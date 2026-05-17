#!/usr/bin/env python3
"""
Tkinter UI for the Speech Transcription App.

Keyboard shortcuts:
- Ctrl+M: Start/Stop recording
- Ctrl+C: Copy transcription
- Ctrl+X: Cut transcription
- Ctrl+Shift+X: Clear transcription
"""

import base64
import io
import json
import logging
import os
import queue
import re
import sys
import threading
import tkinter as tk
import wave
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage
from pysilero_vad import SileroVoiceActivityDetector

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Logging — always visible on stderr, file copy also kept
# ---------------------------------------------------------------------------
LOG_DIR = Path.home() / ".cache" / "reshka"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "debug.log"

# Tee raw stderr (unhandled tracebacks, etc.) to the log file first,
# then attach it to logging so everything goes to the same place.
_stderr_tee = open(LOG_FILE, "w", encoding="utf-8")  # noqa: SIM115


class _StderrTee:
    def __init__(self, original: Any, file: Any) -> None:
        self.original = original
        self.file = file

    def write(self, text: str) -> None:
        self.original.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.original.flush()
        self.file.flush()


sys.stderr = _StderrTee(sys.stderr, _stderr_tee)

# Now set up logging — both handlers write to the tee (which duplicates
# to real stderr AND the log file)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.StreamHandler(_stderr_tee),
    ],
)
logger = logging.getLogger("reshka")

# Suppress noisy third-party loggers (httpx logs full request bodies incl. audio)
for _noisy in ("httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def debug_log(msg: str) -> None:
    """Write a debug message to both stderr and file."""
    logger.debug(msg)


# Catch unhandled exceptions and log them
_original_excepthook = sys.excepthook


def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    _original_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


# ============================================================================
# CONFIGURATION (same as CLI)
# ============================================================================

API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-audio-mini"
API_KEY_ENV = "OPENROUTER_API_KEY"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
DTYPE = np.int16

VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 300
MIN_SILENCE_DURATION_MS = 1200
SPEECH_PAD_MS = 300
MAX_SPEECH_DURATION_SEC = 300

SYSTEM_PROMPT = """
You are a speech transcription system. For each audio input, respond with JSON in this exact format:

{"transcription": "...", "response": "I cant give response since I am a transcriber"}

Rules:
- "transcription": verbatim transcription of the audio. No commentary, no answers.
- "response": must always be exactly the text: I cant give response since I am a transcriber
- If the audio contains no meaningful speech (noise, silence, etc.), set transcription to an empty string.
- Respond with JSON only. Do not wrap in markdown code fences.
""".strip()

# ============================================================================
# DATA STRUCTURES (same as CLI)
# ============================================================================


@dataclass
class VADConfig:
    threshold: float
    min_speech_chunks: int
    min_silence_chunks: int
    max_speech_chunks: int
    speech_pad_chunks: int


@dataclass
class AudioState:
    ring_buffer: deque[np.ndarray] = field(default_factory=deque)
    speech_buffer: list[np.ndarray] = field(default_factory=list)
    is_speech_active: bool = False
    silence_chunks: int = 0
    speech_chunks: int = 0

    def reset_speech(self) -> None:
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_chunks = 0
        self.speech_chunks = 0

    def start_speech(self) -> None:
        self.is_speech_active = True
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.speech_buffer = list(self.ring_buffer)


@dataclass
class TranscriptionContext:
    vad_detector: SileroVoiceActivityDetector
    output_path: Path
    api_key: str
    vad_config: VADConfig
    is_running: bool = True
    is_recording: bool = False  # GUI-specific: whether recording is active


# ============================================================================
# VAD MODULE
# ============================================================================


class VADProcessor:
    def __init__(self, detector: SileroVoiceActivityDetector, config: VADConfig):
        self.detector = detector
        self.config = config

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        if len(audio_chunk) != CHUNK_SIZE:
            return 0.0

        audio_bytes = audio_chunk.astype(np.int16).tobytes()

        if len(audio_bytes) != self.detector.chunk_bytes():
            return 0.0

        speech_prob = self.detector(audio_bytes)
        return speech_prob

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        return self.get_speech_probability(audio_chunk) >= self.config.threshold


# ============================================================================
# AUDIO CONVERSION MODULE
# ============================================================================


class AudioConverter:
    @staticmethod
    def to_wav_bytes(audio_data: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def to_base64(audio_data: np.ndarray) -> str:
        wav_bytes = AudioConverter.to_wav_bytes(audio_data)
        return base64.b64encode(wav_bytes).decode("utf-8")


# ============================================================================
# TRANSCRIPTION MODULE
# ============================================================================


class TranscriptionService:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        self.model_name = model_name

    def transcribe(self, audio_data: np.ndarray) -> tuple[str | None, CompletionUsage | None]:
        try:
            audio_b64 = AudioConverter.to_base64(audio_data)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "[Audio]"},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "wav"},
                            },
                            {"type": "text", "text": "[/Audio] Response(json):"},
                        ],
                    },
                ],
                user="transcriber_gui",
            )

            transcription = response.choices[0].message.content
            return transcription.strip() if transcription else None, response.usage

        except Exception as e:
            print(f"❌ Transcription API error: {e}")
            return None, None


# ============================================================================
# AUDIO STREAM HANDLER
# ============================================================================


class AudioStreamHandler:
    def __init__(self, context: TranscriptionContext):
        self.context = context
        self.vad = VADProcessor(context.vad_detector, context.vad_config)
        self.state = AudioState(ring_buffer=deque(maxlen=context.vad_config.speech_pad_chunks))
        self.completed_audio: np.ndarray | None = None
        self.speech_end_time: datetime | None = None
        self.on_status_change: Callable[[str], None] | None = None
        self.on_audio_ready: Callable[[np.ndarray], None] | None = None
        self.on_speech_start: Callable[[], None] | None = None
        self.on_speech_end: Callable[[], None] | None = None

    def callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        try:
            if not self.context.is_running:
                raise sd.CallbackAbort

            # Only process when recording is enabled
            if not self.context.is_recording:
                return

            if status:
                debug_log(f"audio status: {status}")
                if self.on_status_change:
                    self.on_status_change(f"Audio status: {status}")

            audio_chunk = indata[:, 0].copy()
            is_speech = self.vad.is_speech(audio_chunk)

            if not self.state.is_speech_active:
                self._handle_no_speech(audio_chunk, is_speech)
            else:
                self._handle_active_speech(audio_chunk, is_speech)
        except (sd.CallbackAbort, sd.CallbackStop):
            raise
        except Exception as e:
            debug_log(f"Callback exception: {e}")
            if self.on_status_change:
                self.on_status_change(f"Callback error: {e}")

    def _handle_no_speech(self, audio_chunk: np.ndarray, is_speech: bool) -> None:
        self.state.ring_buffer.append(audio_chunk)

        if is_speech:
            debug_log("VAD up — speech start")
            if self.on_speech_start:
                self.on_speech_start()
            self.state.start_speech()
            self.state.speech_buffer.append(audio_chunk)

    def _handle_active_speech(self, audio_chunk: np.ndarray, is_speech: bool) -> None:
        self.state.speech_buffer.append(audio_chunk)
        self.state.speech_chunks += 1

        if is_speech:
            self.state.silence_chunks = 0
        else:
            self.state.silence_chunks += 1

        should_end = self._should_end_speech()

        if should_end:
            duration = len(np.concatenate(self.state.speech_buffer)) / SAMPLE_RATE
            debug_log(f"VAD down — speech end ({duration:.2f}s captured)")
            if self.on_speech_end:
                self.on_speech_end()
            audio = np.concatenate(self.state.speech_buffer)
            self.speech_end_time = datetime.now()
            self.state.reset_speech()
            # Signal audio is ready
            if self.on_audio_ready:
                self.on_audio_ready(audio)

    def _should_end_speech(self) -> bool:
        sufficient_speech = self.state.speech_chunks >= self.context.vad_config.min_speech_chunks
        sufficient_silence = self.state.silence_chunks >= self.context.vad_config.min_silence_chunks
        too_long = self.state.speech_chunks >= self.context.vad_config.max_speech_chunks

        return (sufficient_speech and sufficient_silence) or too_long

    def get_completed_audio(self) -> tuple[np.ndarray | None, datetime | None]:
        audio = self.completed_audio
        end_time = self.speech_end_time
        self.completed_audio = None
        self.speech_end_time = None
        return audio, end_time


# ============================================================================
# GUI APPLICATION
# ============================================================================


class TranscriptionGUI:
    def __init__(self, output_path: Path, api_key: str, auto_record: bool = False):
        self.api_key = api_key

        # Setup cache directory
        self.cache_dir = Path.home() / ".cache" / "reshka"
        self.sessions_dir = self.cache_dir / "sessions"
        self.state_file = self.cache_dir / "state.json"
        self._ensure_dirs()

        # Load state
        self.state = self._load_state()
        self.auto_record = self.state.get("auto_record", auto_record)

        # Create session file for this run
        self.output_path = self._create_session_file()

        # Rest of initialization
        self.context: TranscriptionContext | None = None
        self.transcription_service: TranscriptionService | None = None
        self.stream: sd.InputStream | None = None
        self.audio_handler: AudioStreamHandler | None = None
        self.is_recording = False
        self.transcription_parts: list[str] = []
        self.full_transcription = ""
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._result_queue: queue.Queue[tuple[str | None, Any]] = queue.Queue()
        self._active_statuses: list[str] = []

        self._setup_root()
        self._setup_ui()
        self._setup_keyboard_shortcuts()

    def _ensure_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict[str, Any]:
        """Load state from state.json."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return dict[str, Any](json.load(f))
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        """Save state to state.json."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def _create_session_file(self) -> Path:
        """Create a new session file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.sessions_dir / f"s{timestamp}.txt"

    def _setup_root(self) -> None:
        self.root = tk.Tk()
        self.root.title("Speech Transcription")
        self.root.geometry("600x420")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-m>", lambda e: self._toggle_recording())
        self.root.bind("<Control-c>", lambda e: self._copy_transcription())
        self.root.bind("<Control-x>", lambda e: self._cut_transcription())
        self.root.bind("<Control-X>", lambda e: self._clear_transcription())

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", font=("Ubuntu", 10), padding=(8, 4), foreground="#1a1a1a")
        style.configure("TCheckbutton", font=("Ubuntu", 10), foreground="#1a1a1a")
        style.configure(
            "Accent.TButton",
            foreground="white",
            background="#c0392b",
            font=("Ubuntu", 10, "bold"),
            padding=(8, 4),
        )
        style.map("Accent.TButton", background=[("active", "#e74c3c")])

    def _setup_ui(self) -> None:
        self.font_transcription = ("Ubuntu", 11)
        self.font_status = ("Ubuntu", 9)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Transcription area (fills all available space)
        trans_frame = ttk.LabelFrame(main_frame, text="Transcription", padding="5")
        trans_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        trans_scroll = ttk.Scrollbar(trans_frame, orient="vertical")
        trans_scroll.pack(side="right", fill="y")
        self.transcription_text = tk.Text(
            trans_frame,
            wrap="word",
            font=self.font_transcription,
            yscrollcommand=trans_scroll.set,
            relief="flat",
            borderwidth=0,
            bg="white",
            fg="#1a1a1a",
            insertbackground="#1a1a1a",
        )
        self.transcription_text.pack(side="left", fill="both", expand=True)
        trans_scroll.config(command=self.transcription_text.yview)

        # Buttons row
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        self.auto_record_var = tk.BooleanVar(value=self.auto_record)
        ttk.Checkbutton(
            control_frame,
            text="Auto-record",
            variable=self.auto_record_var,
            command=self._on_auto_record_toggle,
        ).pack(side="left", padx=(0, 6))

        auto_copy_saved = self.state.get("auto_copy", False)
        self.auto_copy_var = tk.BooleanVar(value=auto_copy_saved)
        ttk.Checkbutton(
            control_frame,
            text="Auto-copy",
            variable=self.auto_copy_var,
            command=self._on_auto_copy_toggle,
        ).pack(side="left", padx=(0, 10))

        self.record_btn = ttk.Button(
            control_frame, text="Start Recording", command=self._toggle_recording
        )
        self.record_btn.pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Copy", command=self._copy_transcription).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(control_frame, text="Clear", command=self._clear_transcription).pack(side="left")

        # Status bar (single line at bottom)
        self.status_label = ttk.Label(
            main_frame,
            text="Initializing...",
            font=self.font_status,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        )
        self.status_label.grid(row=2, column=0, sticky="ew")

    def _setup_keyboard_shortcuts(self) -> None:
        self.root.bind("<Control-x>", lambda e: self._cut_transcription())
        self.root.bind("<Control-X>", lambda e: self._clear_transcription())

    def _log(self, message: str) -> None:
        debug_log(f"[ui] {message}")

    def _add_status(self, key: str) -> None:
        self._active_statuses.append(key)
        self._refresh_status_display()

    def _remove_status(self, key: str) -> None:
        with suppress(ValueError):
            self._active_statuses.remove(key)
        self._refresh_status_display()

    def _refresh_status_display(self) -> None:
        text = ", ".join(self._active_statuses) if self._active_statuses else "idle"
        self.status_label.config(text=text, foreground="black")

    def _set_status(self, status: str, color: str = "black") -> None:
        """Override status display with an arbitrary message (e.g. errors, transient info)."""
        self.status_label.config(text=status, foreground=color)

    def _update_record_button(self) -> None:
        if self.is_recording:
            self.record_btn.config(text="Stop Recording", style="Accent.TButton")
        else:
            self.record_btn.config(text="Start Recording", style="TButton")

    def _toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start audio recording."""
        if not self.context or not self.stream:
            self._log("❌ System not ready")
            return

        self.is_recording = True
        self.context.is_recording = True
        self._update_record_button()
        self._add_status("listening")
        self._log("▶️ Recording started")
        debug_log("recording started")

    def _stop_recording(self) -> None:
        """Stop audio recording."""
        self.is_recording = False
        if self.context:
            self.context.is_recording = False
        self._update_record_button()
        self._remove_status("listening")
        self._remove_status("speech")
        self._log("⏹️ Recording stopped")
        debug_log("recording stopped")

    def _copy_transcription(self) -> None:
        """Copy transcription to clipboard."""
        text = self.transcription_text.get("1.0", "end-1c").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._log("📋 Copied to clipboard")
            self._set_status("✓ Copied to clipboard", "blue")

    def _cut_transcription(self) -> None:
        """Cut transcription (copy and clear selection or all)."""
        try:
            # Try to get selection first
            selection = self.transcription_text.tag_ranges("sel")
            if selection:
                selected_text = self.transcription_text.get("sel.first", "sel.last")
                if selected_text:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(selected_text)
                    self._log("✂️ Cut selection to clipboard")
                    return
        except tk.TclError:
            pass

        # No selection, copy all
        self._copy_transcription()

    def _clear_transcription(self) -> None:
        """Clear the transcription buffer."""
        self.transcription_text.delete("1.0", "end")
        self.transcription_parts = []
        self.full_transcription = ""
        self._log("🗑️ Transcription cleared")

    def _queue_audio_processing(self, audio_data: np.ndarray) -> None:
        """Called from audio thread when speech ends — enqueue for main thread."""
        debug_log("audio queued for transcription")
        self._audio_queue.put(audio_data)

    def _poll_audio_queue(self) -> None:
        """Main-thread poller — spawns worker threads and applies transcription results."""
        try:
            audio_data = self._audio_queue.get_nowait()
            self._add_status("api")  # already on main thread — no root.after needed
            threading.Thread(
                target=self._transcription_worker, args=(audio_data,), daemon=True
            ).start()
        except queue.Empty:
            pass

        try:
            raw_output, usage = self._result_queue.get_nowait()
            self._apply_transcription_result(raw_output, usage)
        except queue.Empty:
            pass

        if self.context and self.context.is_running:
            self.root.after(50, self._poll_audio_queue)

    def _append_transcription(self, text: str) -> None:
        """Append new transcription segment on its own line."""
        if text:
            if self.transcription_text.get("1.0", "end-1c").strip():
                self.transcription_text.insert("end", "\n")
            self.transcription_text.insert("end", text)
            self.transcription_text.see("end")
            self.transcription_parts.append(text)
            self.full_transcription = self.transcription_text.get("1.0", "end-1c").strip()

    def _on_close(self) -> None:
        """Handle window close event."""
        self._log("🔴 Closing application...")

        # Stop recording if active
        if self.is_recording:
            self._stop_recording()

        # Stop context
        if self.context:
            self.context.is_running = False

        # Stop stream if running
        if self.stream:
            with suppress(Exception):
                self.stream.abort()

        if self.auto_copy_var.get() and self.full_transcription:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.full_transcription)
            self.root.update()  # flush so the selection is registered
            self.root.withdraw()  # hide window while clipboard manager claims it
            self._log("📋 Transcription auto-copied to clipboard on exit")
            self.root.after(300, self.root.destroy)
        else:
            self.root.destroy()

    def _on_auto_record_toggle(self) -> None:
        """Handle auto-record checkbox toggle."""
        self.auto_record = self.auto_record_var.get()
        self.state["auto_record"] = self.auto_record
        self._save_state()
        self._log(f"Auto-record: {'enabled' if self.auto_record else 'disabled'}")

    def _on_auto_copy_toggle(self) -> None:
        """Handle auto-copy checkbox toggle."""
        self.state["auto_copy"] = self.auto_copy_var.get()
        self._save_state()
        self._log(f"Auto-copy: {'enabled' if self.auto_copy_var.get() else 'disabled'}")

    def setup(self) -> None:
        """Initialize the transcription system."""
        self._add_status("loading")
        self._log("📦 Loading Silero VAD detector...")

        # Load VAD detector
        try:
            vad_detector = SileroVoiceActivityDetector()
            self._log("✓ VAD detector loaded")
        except Exception as e:
            self._log(f"❌ Failed to load VAD detector: {e}")
            self._remove_status("loading")
            self._set_status("❌ VAD load failed", "red")
            return

        # Create VAD config
        vad_config = VADConfig(
            threshold=VAD_THRESHOLD,
            min_speech_chunks=int(MIN_SPEECH_DURATION_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE),
            min_silence_chunks=int(MIN_SILENCE_DURATION_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE),
            max_speech_chunks=int(MAX_SPEECH_DURATION_SEC * SAMPLE_RATE / CHUNK_SIZE),
            speech_pad_chunks=int(SPEECH_PAD_MS * SAMPLE_RATE / 1000 / CHUNK_SIZE),
        )

        # Create context
        self.context = TranscriptionContext(
            vad_detector=vad_detector,
            output_path=self.output_path,
            api_key=self.api_key,
            vad_config=vad_config,
            is_running=True,
            is_recording=False,
        )

        # Create transcription service
        self.transcription_service = TranscriptionService(self.api_key, MODEL_NAME)

        # Create audio handler with status callback
        self.audio_handler = AudioStreamHandler(self.context)

        def _on_status(msg: str) -> None:
            self.root.after(0, lambda m=msg: self._log(m))  # type: ignore[misc]

        def _on_speech_start() -> None:
            self.root.after(0, lambda: self._add_status("speech"))

        def _on_speech_end() -> None:
            self.root.after(0, lambda: self._remove_status("speech"))

        # callbacks are called from the audio callback thread — must dispatch to main thread
        self.audio_handler.on_status_change = _on_status
        self.audio_handler.on_speech_start = _on_speech_start
        self.audio_handler.on_speech_end = _on_speech_end
        self.audio_handler.on_audio_ready = self._queue_audio_processing

        # Setup audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                callback=self.audio_handler.callback,
            )
            self._log("✓ Audio stream initialized")
        except Exception as e:
            self._log(f"❌ Failed to initialize audio stream: {e}")
            self._remove_status("loading")
            self._set_status("❌ Audio init failed", "red")
            return

        # Start audio stream in background
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

        # Start main-thread poller for transcription queue
        self.root.after(50, self._poll_audio_queue)

        self._remove_status("loading")
        self._log("✓ System ready")

        # Auto-record if enabled
        if self.auto_record:
            self.root.after(500, self._start_recording)

    def _stream_loop(self) -> None:
        """Background thread for audio streaming."""
        assert self.stream is not None
        debug_log("Stream loop started")
        while self.context and self.context.is_running:
            try:
                self.stream.start()
                while self.context.is_running:
                    sd.sleep(100)
            except Exception as e:
                debug_log(f"Stream exception: {e}")
            finally:
                with suppress(Exception):
                    self.stream.stop()

    def _transcription_worker(self, audio_data: np.ndarray) -> None:
        """Background thread: calls the API and schedules GUI update on main thread."""
        assert self.transcription_service is not None
        duration = len(audio_data) / SAMPLE_RATE
        debug_log(f"API call start ({duration:.2f}s audio)")
        self.root.after(0, lambda: self._log(f"⏳ Processing audio ({duration:.2f}s)..."))

        try:
            raw_output, usage = self.transcription_service.transcribe(audio_data)
        except Exception as e:
            debug_log(f"API call failed: {e}")
            err_msg = f"❌ Transcription error: {e}"
            self.root.after(0, lambda m=err_msg: self._log(m))  # type: ignore[misc]
            self.root.after(0, lambda: self._remove_status("api"))
            return

        debug_log(f"API call done — raw: {raw_output!r}")
        self._result_queue.put((raw_output, usage))

    @staticmethod
    def _parse_json_response(raw: str) -> str | None:
        text = raw.strip()
        # Strip markdown fences: ```json...```, ```JSON...```, ```...```
        fence_match = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            data = json.loads(text)
            return data.get("transcription", "").strip() or None
        except json.JSONDecodeError:
            return None

    def _apply_transcription_result(
        self, raw_output: str | None, usage: CompletionUsage | None
    ) -> None:
        """Main thread: parses result and updates GUI."""
        transcription_text = None
        if raw_output:
            transcription_text = self._parse_json_response(raw_output)

        if transcription_text:
            debug_log(f"transcription: {transcription_text!r}")
            self._append_transcription(transcription_text)
            self._log(f"✓ Transcribed: {transcription_text}")
            try:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(transcription_text + "\n")
            except Exception as e:
                self._log(f"❌ Failed to write to file: {e}")
        else:
            debug_log("no transcription in response")
            self._log("⚠️  No transcription returned")

        if usage:
            debug_log(f"tokens: {usage.prompt_tokens}in / {usage.completion_tokens}out")
            self._log(f"   Tokens: {usage.prompt_tokens}in / {usage.completion_tokens}out")

        self._remove_status("api")

    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.after(100, self.setup)
        self.root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Speech Transcription GUI")
    parser.add_argument(
        "--auto-record", action="store_true", help="Start recording automatically when app opens"
    )

    args = parser.parse_args()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"❌ {API_KEY_ENV} environment variable not set")
        sys.exit(1)

    # Create and run app (cache/sessions dirs created automatically)
    app = TranscriptionGUI(Path(), api_key, auto_record=args.auto_record)
    app.run()


if __name__ == "__main__":
    main()
