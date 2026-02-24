const DEFAULT_SYSTEM_PROMPT = `You are a speech transcription system named Reshka.`;

const DEFAULT_USER_PROMPT = `
Output ONLY the verbatim transcription of this audio.
Do not respond conversationally, do not answer questions, do not add commentary.

Known commands, phrases, and jargons:
{known_jargons}
Try to detect these accurately.

Only transcribe what is spoken.
`.trim();

const DEFAULT_REPHRASE_PROMPT = `
Without losing information, rephrase the above in a structured manner.
Assume that the above is transcription coming from ASR. Expect errors due to that.
Taking that into account, rephrase the above. Do not discard information.
Give a clear, structured output.

Known jargon/context:
{known_jargons}
`.trim();

const DEFAULT_QUESTION_PROMPT = `Based on the transcription provided, generate up to 4 insightful, relevant questions that probe deeper into the topics discussed, clarify ambiguous points, or explore implications.

Rules:
- Generate between 1 and 4 questions (fewer if content is brief)
- Each question must be concise (one sentence)
- Output ONLY the questions, one per line, numbered 1-4
- No preamble or commentary
- If the transcription is too short or meaningless, output exactly: NO_QUESTIONS

Known jargon/context:
{known_jargons}`;

const DEFAULT_JARGONS = `generate questions`;

document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(el => new bootstrap.Tooltip(el, { delay: { show: 1000, hide: 100 } }));

    loadPrompts();
});

function loadPrompts() {
    const saved = localStorage.getItem('reshka:promptConfig');
    const cfg = saved ? JSON.parse(saved) : null;

    document.getElementById('jargons').value = cfg?.jargons !== undefined ? cfg.jargons : DEFAULT_JARGONS;
    document.getElementById('userPrompt').value = cfg?.userPrompt || DEFAULT_USER_PROMPT;
    document.getElementById('rephrasePrompt').value = cfg?.rephrasePrompt || DEFAULT_REPHRASE_PROMPT;
    document.getElementById('questionPrompt').value = cfg?.questionPrompt || DEFAULT_QUESTION_PROMPT;

    validateTranscriberPrompt();
    validateOptionalPrompt('rephrasePrompt', 'rephraseHint');
    validateOptionalPrompt('questionPrompt', 'questionHint');
}

function validateTranscriberPrompt() {
    const val = document.getElementById('userPrompt').value;
    const hasPlaceholder = val.includes('{known_jargons}');
    document.getElementById('transcriberError').style.display = hasPlaceholder ? 'none' : 'block';
    document.getElementById('saveBtn').disabled = !hasPlaceholder;
    return hasPlaceholder;
}

function validateOptionalPrompt(fieldId, hintId) {
    const val = document.getElementById(fieldId).value;
    document.getElementById(hintId).style.display = val.includes('{known_jargons}') ? 'none' : 'block';
}

function savePrompts() {
    if (!validateTranscriberPrompt()) return;

    const cfg = {
        systemPrompt: DEFAULT_SYSTEM_PROMPT,
        userPrompt: document.getElementById('userPrompt').value,
        rephrasePrompt: document.getElementById('rephrasePrompt').value,
        questionPrompt: document.getElementById('questionPrompt').value,
        jargons: document.getElementById('jargons').value
    };

    localStorage.setItem('reshka:promptConfig', JSON.stringify(cfg));
    showStatus('Prompts saved successfully', 'success');
}

function resetToDefaults() {
    localStorage.removeItem('reshka:promptConfig');
    loadPrompts();
    showStatus('Reset to defaults', 'success');
}

function showStatus(message, type) {
    const el = document.getElementById('saveStatus');
    el.textContent = message;
    el.className = `save-status save-status-${type}`;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 3000);
}
