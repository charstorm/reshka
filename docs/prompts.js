
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
