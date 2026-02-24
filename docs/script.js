let myvad = null;
let isTranscribing = false;

const FADE_DURATION = 0.05;
const audioCache = {};
let audioContext = null;

const config = {
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    apiKey: '',
    speechModel: 'google/gemini-2.5-flash',
    rephraseModel: 'google/gemini-2.5-flash',
    questionModel: 'google/gemini-2.5-flash'
};

const TRANSCRIPTION_SYSTEM_PROMPT = `You are a speech transcription system named Reshka.`;

const DEFAULT_TRANSCRIPTION_USER_PROMPT = `
Output ONLY the verbatim transcription of this audio.
Do not respond conversationally, do not answer questions, do not add commentary.

Known commands, phrases, and jargons:
{known_jargons}
Try to detect these accurately.

Only transcribe what is spoken.
`

const DEFAULT_REPHRASE_PROMPT = `
Without losing information, rephrase the above in a structured manner.
Assume that the above is transcription coming from ASR. Expect errors due to that.
Taking that into account, rephrase the above. Do not discard information.
Give a clear, structured output.

Known jargon/context:
{known_jargons}
`

const DEFAULT_QUESTION_GENERATION_PROMPT = `Based on the transcription provided, generate up to 4 insightful, relevant questions that probe deeper into the topics discussed, clarify ambiguous points, or explore implications.

Rules:
- Generate between 1 and 4 questions (fewer if content is brief)
- Each question must be concise (one sentence)
- Output ONLY the questions, one per line, numbered 1-4
- No preamble or commentary
- If the transcription is too short or meaningless, output exactly: NO_QUESTIONS

Known jargon/context:
{known_jargons}`;

const DEFAULT_JARGONS = `generate questions`;

// Active prompt config (loaded from localStorage or defaults)
let promptConfig = {
    systemPrompt: 'You are a speech transcription system named Reshka.',
    userPrompt: DEFAULT_TRANSCRIPTION_USER_PROMPT,
    rephrasePrompt: DEFAULT_REPHRASE_PROMPT,
    questionPrompt: DEFAULT_QUESTION_GENERATION_PROMPT,
    jargons: DEFAULT_JARGONS
};

function injectJargons(prompt, jargons) {
    const lines = jargons.split('\n').filter(l => l.trim()).map(l => `- "${l.trim()}"`).join('\n');
    return prompt.replace('{known_jargons}', lines || '(none)');
}

const QUESTION_COMMAND_PATTERN = /^(?:please\s+)?generate\s+questions(?:[,.]?\s*please)?[.!]?$/i;

const VAD_CONFIG = {
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    redemptionFrames: 8,
    minSpeechFrames: 15
};

const rephraseModal = new bootstrap.Modal(document.getElementById('rephraseModal'));

document.addEventListener('DOMContentLoaded', async () => {
    initTooltips();
    loadConfig();
    loadPromptConfig();
    loadTranscript();
    loadRephraseResult();
    loadQuestions();

    const toggleBtn = document.getElementById('toggleBtn');
    toggleBtn.disabled = true;
    addLog('Generating audio feedback...', 'info');
    await generateAllSounds();
    addLog('Setup complete', 'success');
    toggleBtn.disabled = false;
});

function initTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(el => new bootstrap.Tooltip(el, {
        delay: { show: 1000, hide: 100 }
    }));
}

function loadConfig() {
    const savedConfig = localStorage.getItem('reshka:appConfig');
    if (savedConfig) {
        const parsed = JSON.parse(savedConfig);
        config.endpoint = parsed.endpoint || config.endpoint;
        config.apiKey = parsed.apiKey || config.apiKey;
        config.speechModel = parsed.speechModel || config.speechModel;
        config.rephraseModel = parsed.rephraseModel || config.rephraseModel;
        config.questionModel = parsed.questionModel || config.questionModel;
    }
}

function loadPromptConfig() {
    const saved = localStorage.getItem('reshka:promptConfig');
    if (saved) {
        const parsed = JSON.parse(saved);
        promptConfig.systemPrompt = parsed.systemPrompt || promptConfig.systemPrompt;
        promptConfig.userPrompt = parsed.userPrompt || promptConfig.userPrompt;
        promptConfig.rephrasePrompt = parsed.rephrasePrompt || promptConfig.rephrasePrompt;
        promptConfig.questionPrompt = parsed.questionPrompt || promptConfig.questionPrompt;
        promptConfig.jargons = parsed.jargons !== undefined ? parsed.jargons : promptConfig.jargons;
    }
}

async function validatePrerequisites() {
    addLog('Starting pre-flight validation...', 'info');
    updateStatus('processing', 'Validating...');

    const apiKeyValid = await checkApiKey();
    if (!apiKeyValid) {
        updateStatus('ready', 'Ready');
        return false;
    }

    const audioPermissionValid = await checkAudioPermissions();
    if (!audioPermissionValid) {
        updateStatus('ready', 'Ready');
        return false;
    }

    addLog('All validations passed', 'success');
    updateStatus('ready', 'Ready');
    return true;
}

async function checkApiKey() {
    addLog('Checking API key configuration...', 'info');

    if (!config.apiKey) {
        addLog('ERROR: API key not configured. Redirecting to config...', 'error');
        showToast('Please configure your API key first', 'error');
        setTimeout(() => { window.location.href = 'config.html'; }, 1500);
        return false;
    }

    addLog('Verifying API key with OpenRouter...', 'info');

    try {
        const response = await fetch('https://openrouter.ai/api/v1/auth/key', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${config.apiKey}`
            }
        });

        if (!response.ok) {
            addLog(`ERROR: API key verification failed (${response.status})`, 'error');
            showToast('Invalid API key. Please check configuration.', 'error');
            setTimeout(() => { window.location.href = 'config.html'; }, 1500);
            return false;
        }

        addLog('API key verified successfully', 'success');
        return true;
    } catch (error) {
        addLog(`ERROR: API key check failed: ${error.message}`, 'error');
        showToast('Failed to verify API key. Check internet connection.', 'error');
        return false;
    }
}

async function checkAudioPermissions() {
    addLog('Checking microphone permissions...', 'info');

    try {
        const permissions = await navigator.permissions.query({ name: 'microphone' });

        if (permissions.state === 'granted') {
            addLog('Microphone permission already granted', 'success');
            return true;
        } else if (permissions.state === 'prompt') {
            addLog('Requesting microphone access...', 'info');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            addLog('Microphone access granted', 'success');
            return true;
        } else {
            addLog('ERROR: Microphone permission denied', 'error');
            showToast('Microphone access denied. Please enable permissions in browser settings.', 'error');
            return false;
        }
    } catch (error) {
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            addLog('ERROR: Microphone permission denied', 'error');
            showToast('Microphone access denied. Please enable permissions.', 'error');
        } else {
            addLog(`ERROR: Microphone check failed: ${error.message}`, 'error');
            showToast('Failed to check microphone: ' + error.message, 'error');
        }
        return false;
    }
}

function clearActivityLog() {
    const logArea = document.getElementById('logArea');
    logArea.innerHTML = '';
}


function float32ToWav(float32Array, sampleRate = 16000) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const buffer = new ArrayBuffer(44 + int16Array.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + int16Array.length * 2, true);
    writeString(view, 8, 'WAVE');

    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);

    writeString(view, 36, 'data');
    view.setUint32(40, int16Array.length * 2, true);

    const bytes = new Uint8Array(buffer, 44);
    for (let i = 0; i < int16Array.length; i++) {
        bytes[i * 2] = int16Array[i] & 0xFF;
        bytes[i * 2 + 1] = (int16Array[i] >> 8) & 0xFF;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

async function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

function createWavBlob(audioBuffer) {
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length;
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    const channelData = audioBuffer.getChannelData(0);

    const writeStr = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, length * 2, true);

    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function generateVadUp(duration, fadeOut = FADE_DURATION) {
    const ctx = initAudioContext();
    const sampleRate = ctx.sampleRate;
    const numSamples = Math.floor(sampleRate * duration);
    const audioBuffer = ctx.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    const fadeOutSamples = Math.floor(sampleRate * fadeOut);

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        let amplitude = 0.05;
        if (i > numSamples - fadeOutSamples) {
            amplitude *= (numSamples - i) / fadeOutSamples;
        }
        channelData[i] = amplitude * Math.sin(2 * Math.PI * 700 * t);
    }

    return audioBuffer;
}

function generateVadDown(duration, fadeOut = FADE_DURATION) {
    const ctx = initAudioContext();
    const sampleRate = ctx.sampleRate;
    const numSamples = Math.floor(sampleRate * duration);
    const audioBuffer = ctx.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    const fadeOutSamples = Math.floor(sampleRate * fadeOut);

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        let amplitude = 0.3;
        if (i > numSamples - fadeOutSamples) {
            amplitude *= (numSamples - i) / fadeOutSamples;
        }
        channelData[i] = amplitude * Math.sin(2 * Math.PI * 500 * t);
    }

    return audioBuffer;
}

function generateChime(duration, fadeOut = FADE_DURATION) {
    const ctx = initAudioContext();
    const sampleRate = ctx.sampleRate;
    const numSamples = Math.floor(sampleRate * duration);
    const audioBuffer = ctx.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    const fadeOutSamples = Math.floor(sampleRate * fadeOut);
    const halfDuration = duration / 2;

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const frequency = t < halfDuration ? 600 : 800;
        let amplitude = 0.3;
        if (i > numSamples - fadeOutSamples) {
            amplitude *= (numSamples - i) / fadeOutSamples;
        }
        channelData[i] = amplitude * Math.sin(2 * Math.PI * frequency * t);
    }

    return audioBuffer;
}

function generateError(duration, fadeOut = FADE_DURATION) {
    const ctx = initAudioContext();
    const sampleRate = ctx.sampleRate;
    const numSamples = Math.floor(sampleRate * duration);
    const audioBuffer = ctx.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    const fadeOutSamples = Math.floor(sampleRate * fadeOut);
    const beepDuration = duration / 3;

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const segment = Math.floor(t / beepDuration);
        let amplitude = segment < 3 ? 0.5 : 0;
        const segmentT = t % beepDuration;
        if (segmentT > beepDuration * 0.4) {
            amplitude = 0;
        }
        if (i > numSamples - fadeOutSamples) {
            amplitude *= (numSamples - i) / fadeOutSamples;
        }
        channelData[i] = amplitude * Math.sin(2 * Math.PI * 400 * t);
    }

    return audioBuffer;
}

function generateMisfire(duration, fadeOut = FADE_DURATION) {
    const ctx = initAudioContext();
    const sampleRate = ctx.sampleRate;
    const numSamples = Math.floor(sampleRate * duration);
    const audioBuffer = ctx.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    const fadeOutSamples = Math.floor(sampleRate * fadeOut);

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const frequency = 500 - 200 * (t / duration);
        let amplitude = 0.2;
        if (i > numSamples - fadeOutSamples) {
            amplitude *= (numSamples - i) / fadeOutSamples;
        }
        channelData[i] = amplitude * Math.sin(2 * Math.PI * frequency * t);
    }

    return audioBuffer;
}

async function generateAllSounds() {
    initAudioContext();
    audioCache.vadUp = createWavBlob(generateVadUp(0.2));
    audioCache.vadDown = createWavBlob(generateVadDown(0.2));
    audioCache.transcriptionArrived = createWavBlob(generateChime(0.2));
    audioCache.error = createWavBlob(generateError(0.2));
    audioCache.misfire = createWavBlob(generateMisfire(0.15));
}

function playSound(soundName, onEnded = null) {
    const blob = audioCache[soundName];
    if (!blob) return;

    const audio = new Audio(URL.createObjectURL(blob));
    if (onEnded) {
        audio.onended = onEnded;
    }
    audio.play();
}

async function toggleTranscription() {
    if (isTranscribing) {
        await stopTranscription();
    } else {
        const isValid = await validatePrerequisites();
        if (isValid) {
            await startTranscription();
        }
    }
}

async function startTranscription() {
    if (!config.apiKey) {
        showToast('Please configure your API key first', 'error');
        setTimeout(() => { window.location.href = 'config.html'; }, 1500);
        return;
    }

    try {
        addLog('Initializing Silero VAD...', 'info');
        updateStatus('active', 'Initializing...');

        myvad = await vad.MicVAD.new({
            onSpeechStart: () => {
                updateStatus('speaking', 'Speaking detected');
                addLog('Speech started', 'speech-start');
                playSound('vadUp');
            },

            onSpeechEnd: async (audio) => {
                playSound('vadDown');
                updateStatus('processing', 'Processing...');
                const durationMs = (audio.length / 16000 * 1000).toFixed(0);
                addLog(`Speech ended (${durationMs}ms)`, 'speech-end');

                await processSpeechAudio(audio);

                if (isTranscribing) {
                    updateStatus('active', 'Listening...');
                }
            },

            onVADMisfire: () => {
                addLog('VAD misfire detected', 'warning');
                playSound('misfire');
            },

            positiveSpeechThreshold: VAD_CONFIG.positiveSpeechThreshold,
            negativeSpeechThreshold: VAD_CONFIG.negativeSpeechThreshold,
            redemptionFrames: VAD_CONFIG.redemptionFrames,
            minSpeechFrames: VAD_CONFIG.minSpeechFrames,

            onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
            baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@latest/dist/"
        });

        myvad.start();
        isTranscribing = true;
        updateToggleButton();
        updateStatus('active', 'Listening...');
        addLog('Live transcription started', 'success');
        showToast('Live transcription started');
    } catch (error) {
        console.error('Error starting transcription:', error);
        addLog(`Error: ${error.message}`, 'error');
        showToast('Failed to start transcription: ' + error.message, 'error');
        updateStatus('ready', 'Ready');
        isTranscribing = false;
        updateToggleButton();
    }
}

async function stopTranscription() {
    if (myvad) {
        try {
            await myvad.pause();
            myvad = null;
        } catch (error) {
            console.error('Error stopping VAD:', error);
        }
    }

    isTranscribing = false;
    updateToggleButton();
    updateStatus('ready', 'Ready');
    addLog('Live transcription stopped', 'info');
    showToast('Live transcription stopped');
}

function updateToggleButton() {
    const btn = document.getElementById('toggleBtn');
    if (isTranscribing) {
        btn.innerHTML = '<i class="fas fa-stop me-2"></i>Stop';
        btn.classList.add('active');
    } else {
        btn.innerHTML = '<i class="fas fa-play me-2"></i>Start';
        btn.classList.remove('active');
    }
}

function updateStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    dot.className = 'status-dot ' + state;
    statusText.textContent = text;
}

async function processSpeechAudio(audio) {
    try {
        addLog('Converting audio to WAV...', 'info');

        const wavBlob = float32ToWav(audio, 16000);
        const base64Audio = await blobToBase64(wavBlob);

        addLog('Sending to API...', 'api-call');
        const apiStartTime = performance.now();

        const activeUserPrompt = injectJargons(promptConfig.userPrompt, promptConfig.jargons);
        const requestBody = {
            model: config.speechModel,
            temperature: 0.7,
            messages: [
                {
                    role: 'system',
                    content: promptConfig.systemPrompt
                },
                {
                    role: 'user',
                    content: [
                        {
                            type: 'input_audio',
                            input_audio: {
                                data: base64Audio,
                                format: 'wav'
                            }
                        },
                        {
                            type: 'text',
                            text: activeUserPrompt
                        }
                    ]
                }
            ]
        };

        const response = await fetch(config.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const transcription = data.choices?.[0]?.message?.content ||
            data.content?.[0]?.text ||
            'No transcription available';

        const cleanedTranscription = transcription
            .replace(/```json\n?/gi, '')
            .replace(/```\n?/g, '')
            .trim();

        playSound('transcriptionArrived');
        const elapsed = ((performance.now() - apiStartTime) / 1000).toFixed(1);
        addLog(`Transcription received (${cleanedTranscription.length} chars, ${elapsed}s)`, 'success');
        addTranscriptEntry(cleanedTranscription);

    } catch (error) {
        playSound('error');
        console.error('Error processing speech:', error);
        addLog(`Transcription error: ${error.message}`, 'error');
        showToast('Transcription failed: ' + error.message, 'error');
    }
}

function addLog(message, type = 'info') {
    const logArea = document.getElementById('logArea');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const now = new Date();
    const timestamp = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
    entry.innerHTML = `
        <div class="log-timestamp">${timestamp}</div>
        ${message}
    `;

    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

function addTranscriptEntry(text) {
    // Check for voice commands first — if matched, don't add to transcript
    if (checkForVoiceCommands(text)) {
        return;
    }

    const transcriptArea = document.getElementById('transcriptArea');
    let content = transcriptArea.textContent.trim();
    content += '\n' + text.trim();
    transcriptArea.textContent = content;
    transcriptArea.scrollTop = transcriptArea.scrollHeight;

    saveTranscript();
}


function clearTranscript() {
    const transcriptArea = document.getElementById('transcriptArea');
    transcriptArea.textContent = '';
    localStorage.removeItem('reshka:audioTranscript');
    addLog('Transcript cleared', 'info');
    showToast('Transcript cleared');
}

function copyTranscript() {
    const transcriptArea = document.getElementById('transcriptArea');
    const text = transcriptArea.textContent.trim();

    if (text === '') {
        showToast('No transcript to copy', 'error');
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast('Transcript copied to clipboard');
        addLog('Transcript copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy transcript', 'error');
    });
}

function loadTranscript() {
    const saved = localStorage.getItem('reshka:audioTranscript');
    if (saved) {
        document.getElementById('transcriptArea').textContent = JSON.parse(saved).text;
    }
}

function saveTranscript() {
    const text = document.getElementById('transcriptArea').textContent;
    localStorage.setItem('reshka:audioTranscript', JSON.stringify({ text }));
}

async function rephraseTranscript() {
    const transcriptArea = document.getElementById('transcriptArea');
    const text = transcriptArea.textContent.trim();

    if (text === '') {
        showToast('No transcript to rephrase', 'error');
        return;
    }

    if (!config.apiKey) {
        showToast('Please configure your API key first', 'error');
        setTimeout(() => { window.location.href = 'config.html'; }, 1500);
        return;
    }

    addLog('Rephrasing transcript...', 'api-call');
    updateStatus('processing', 'Rephrasing...');

    try {
        const response = await fetch(config.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            },
            body: JSON.stringify({
                model: config.rephraseModel,
                temperature: 0.7,
                messages: [
                    {
                        role: 'user',
                        content: `${text}\n---\n${injectJargons(promptConfig.rephrasePrompt, promptConfig.jargons)}`
                    }
                ]
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const rephrased = data.choices?.[0]?.message?.content ||
            data.content?.[0]?.text ||
            'No rephrase available';

        saveRephraseResult(rephrased);
        rephraseModal.show();
        updateStatus(isTranscribing ? 'active' : 'ready', isTranscribing ? 'Listening...' : 'Ready');
        addLog('Rephrase completed', 'success');
        showToast('Rephrase completed');
    } catch (error) {
        console.error('Error rephrasing:', error);
        addLog(`Rephrase error: ${error.message}`, 'error');
        showToast('Failed to rephrase: ' + error.message, 'error');
        updateStatus(isTranscribing ? 'active' : 'ready', isTranscribing ? 'Listening...' : 'Ready');
    }
}

function copyRephrase() {
    const rephraseRawContent = document.getElementById('rephraseRawContent');
    const text = rephraseRawContent.textContent;

    if (!text || text.includes('No rephrased content yet')) {
        showToast('No rephrased content to copy', 'error');
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast('Rephrased content copied to clipboard');
        addLog('Rephrased content copied', 'success');
    }).catch(() => {
        showToast('Failed to copy rephrased content', 'error');
    });
}

function showRephraseModal() {
    rephraseModal.show();
}

function loadRephraseResult() {
    const savedRephrase = localStorage.getItem('reshka:rephraseResult');
    if (savedRephrase) {
        document.getElementById('rephraseRawContent').textContent = savedRephrase;
        document.getElementById('rephraseContent').innerHTML = marked.parse(savedRephrase);
    }
}

function saveRephraseResult(content) {
    localStorage.setItem('reshka:rephraseResult', content);
    document.getElementById('rephraseRawContent').textContent = content;
    document.getElementById('rephraseContent').innerHTML = marked.parse(content);
}

async function generateQuestions() {
    const transcriptArea = document.getElementById('transcriptArea');
    const text = transcriptArea.textContent.trim();

    if (text === '') {
        showToast('No transcript to generate questions from', 'error');
        return;
    }

    if (!config.apiKey) {
        showToast('Please configure your API key first', 'error');
        setTimeout(() => { window.location.href = 'config.html'; }, 1500);
        return;
    }

    addLog('Generating questions...', 'api-call');
    updateStatus('processing', 'Generating questions...');

    try {
        const response = await fetch(config.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            },
            body: JSON.stringify({
                model: config.questionModel,
                temperature: 0.7,
                messages: [
                    {
                        role: 'system',
                        content: injectJargons(promptConfig.questionPrompt, promptConfig.jargons)
                    },
                    {
                        role: 'user',
                        content: text
                    }
                ]
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const rawText = data.choices?.[0]?.message?.content ||
            data.content?.[0]?.text ||
            'NO_QUESTIONS';

        if (rawText.trim() === 'NO_QUESTIONS') {
            showToast('Transcript too brief to generate questions', 'error');
            addLog('Transcript too brief for questions', 'info');
        } else {
            displayQuestions(rawText);
            saveQuestions(rawText);
            playSound('transcriptionArrived');
            addLog('Questions generated', 'success');
            showToast('Questions generated');
        }

        updateStatus(isTranscribing ? 'active' : 'ready', isTranscribing ? 'Listening...' : 'Ready');
    } catch (error) {
        playSound('error');
        console.error('Error generating questions:', error);
        addLog(`Question generation error: ${error.message}`, 'error');
        showToast('Failed to generate questions: ' + error.message, 'error');
        updateStatus(isTranscribing ? 'active' : 'ready', isTranscribing ? 'Listening...' : 'Ready');
    }
}

function displayQuestions(rawText) {
    const questionsArea = document.getElementById('questionsArea');
    questionsArea.innerHTML = '';

    const lines = rawText.trim().split('\n').filter(line => line.trim());
    lines.forEach(line => {
        const match = line.match(/^(\d+)[.)]\s*(.*)/);
        const item = document.createElement('div');
        item.className = 'question-item';
        if (match) {
            item.innerHTML = `<span class="question-number">Q${match[1]}</span>${match[2]}`;
        } else {
            item.textContent = line.trim();
        }
        questionsArea.appendChild(item);
    });
}

function clearQuestions() {
    const questionsArea = document.getElementById('questionsArea');
    questionsArea.innerHTML = '<div class="text-muted fst-italic" style="font-size: 0.85rem;">No questions generated yet. Click ⚡ or say "generate questions".</div>';
    localStorage.removeItem('reshka:questions');
    addLog('Questions cleared', 'info');
}

function saveQuestions(content) {
    localStorage.setItem('reshka:questions', content);
}

function loadQuestions() {
    const saved = localStorage.getItem('reshka:questions');
    if (saved) {
        displayQuestions(saved);
    }
}

function checkForVoiceCommands(text) {
    if (QUESTION_COMMAND_PATTERN.test(text.trim())) {
        addLog('Voice command detected: generate questions', 'info');
        generateQuestions();
        return true;
    }
    return false;
}

function showToast(message, type = 'success') {
    addLog(message, type === 'error' ? 'error' : 'success');
}
