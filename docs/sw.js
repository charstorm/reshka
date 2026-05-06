const CACHE = 'reshka-v1';
const SHELL = [
    './',
    './index.html',
    './config.html',
    './prompts.html',
    './style.css',
    './script.js',
    './config.js',
    './prompts.js',
    './manifest.json',
    './icon.svg',
];

self.addEventListener('install', e =>
    e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)))
);

self.addEventListener('fetch', e => {
    if (e.request.url.startsWith(self.location.origin)) {
        e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
    }
});
