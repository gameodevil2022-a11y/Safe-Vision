console.log("SafeVision Background Service Worker Loaded.");

const TEXT_API_URL        = 'http://127.0.0.1:5000/predict_text';
const IMAGE_API_URL       = 'http://127.0.0.1:5000/predict_image';
const IMAGE_BATCH_API_URL = 'http://127.0.0.1:5000/predict_images_batch';


chrome.alarms.create('keepAlive', { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'keepAlive') {
        // Just waking up the service worker
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    
    // --- HANDLER 1: TEXT ---
    if (message.action === "checkText") {
        fetch(TEXT_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'text': message.text })
        })
        .then(res => res.json())
        .then(data => sendResponse(data))
        .catch(err => sendResponse({ status: 'error', reason: err.message }));
        return true; 
    }

    // --- HANDLER 2: IMAGES ---
    if (message.action === "checkImage") {
        // Build request body — supports both URL and base64 imageData
        const body = {};
        if (message.imageData) {
            body.imageData = message.imageData;  // base64 for blob: URLs
        } else {
            body.url = message.url;  // regular URL
        }
        
        fetch(IMAGE_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        })
        .then(res => res.json())
        .then(data => sendResponse(data))
        .catch(err => sendResponse({ status: 'error', reason: err.message }));
        return true;
    }

    // --- HANDLER 3: IMAGE BATCH (faster — one GPU forward pass for all) ---
    if (message.action === "checkImageBatch") {
        fetch(IMAGE_BATCH_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ images: message.images })
        })
        .then(res => res.json())
        .then(data => sendResponse(data))
        .catch(err => sendResponse({ results: [] }));
        return true;
    }

    // --- HANDLER 4: PING (health check from content script) ---
    if (message.action === "ping") {
        sendResponse({ status: 'alive' });
        return false;
    }
});