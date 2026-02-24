const DEFAULTS = {
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    apiKey: '',
    speechModel: 'google/gemini-2.5-flash',
    rephraseModel: 'google/gemini-2.5-flash',
    questionModel: 'google/gemini-2.5-flash',
    autoSleepSeconds: 60
};

document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(el => new bootstrap.Tooltip(el, { delay: { show: 1000, hide: 100 } }));

    loadConfig();
});

function loadConfig() {
    const saved = localStorage.getItem('reshka:appConfig');
    const cfg = saved ? { ...DEFAULTS, ...JSON.parse(saved) } : { ...DEFAULTS };

    document.getElementById('endpointUrl').value = cfg.endpoint;
    document.getElementById('apiKey').value = cfg.apiKey;
    document.getElementById('speechModel').value = cfg.speechModel;
    document.getElementById('rephraseModel').value = cfg.rephraseModel;
    document.getElementById('questionModel').value = cfg.questionModel;
    document.getElementById('autoSleepSeconds').value = cfg.autoSleepSeconds;
}

function saveConfig() {
    const autoSleepRaw = parseInt(document.getElementById('autoSleepSeconds').value, 10);
    const cfg = {
        endpoint: document.getElementById('endpointUrl').value.trim() || DEFAULTS.endpoint,
        apiKey: document.getElementById('apiKey').value.trim(),
        speechModel: document.getElementById('speechModel').value.trim() || DEFAULTS.speechModel,
        rephraseModel: document.getElementById('rephraseModel').value.trim() || DEFAULTS.rephraseModel,
        questionModel: document.getElementById('questionModel').value.trim() || DEFAULTS.questionModel,
        autoSleepSeconds: (!isNaN(autoSleepRaw) && autoSleepRaw >= 10) ? autoSleepRaw : DEFAULTS.autoSleepSeconds
    };

    localStorage.setItem('reshka:appConfig', JSON.stringify(cfg));
    showStatus('Configuration saved successfully', 'success');
}

function showStatus(message, type) {
    const el = document.getElementById('saveStatus');
    el.textContent = message;
    el.className = `save-status save-status-${type}`;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 3000);
}
