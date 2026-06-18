console.log("SafeVision Content Moderator Loaded.");

// ========================================
// CONFIG
// ========================================
const MIN_TEXT_LENGTH = 3;
const PROCESSED_ATTR = 'data-safevision';
const BATCH_SIZE = 5;      // Text: process 5 elements at a time
const BATCH_DELAY = 200;   // Wait 200ms between text batches

// Image batching — collect images for this window then send all at once
const IMG_BATCH_WINDOW_MS = 300;  // collect images for 300ms before sending

// ========================================
// INJECT CSS (survives React re-renders)
// ========================================
const style = document.createElement('style');
style.textContent = `
  .safevision-blur {
    filter: blur(5px) !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    pointer-events: none !important;
    transition: filter 0.3s ease;
  }
  .safevision-blur-img {
    filter: blur(20px) !important;
    pointer-events: none !important;
  }
`;
(document.head || document.documentElement).appendChild(style);

// ========================================
// SKIP LIST — never send these to the API
// ========================================
const SKIP_WORDS = new Set([
    'react', 'reply', 'forward', 'copy', 'pin', 'star', 'delete', 'edit',
    'mute', 'archive', 'unread', 'read', 'block', 'report', 'search',
    'settings', 'logout', 'close', 'cancel', 'save', 'send', 'attach',
    'photo', 'video', 'audio', 'document', 'contact', 'location', 'poll',
    'pdf', 'apk', 'doc', 'jpg', 'png', 'gif', 'mp4', 'mp3',
    'today', 'yesterday', 'monday', 'tuesday', 'wednesday', 'thursday',
    'friday', 'saturday', 'sunday', 'online', 'offline', 'typing',
    'type a message', 'message yourself', 'ask meta ai', 'download',
    'message info', 'get whatsapp for windows', 'end-to-end encrypted',
    '(you)', '1 page', 'unread', 'all', 'groups', 'favourites',
]);

// Tags whose text content should NEVER be scanned
const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'SVG', 'CODE', 'INPUT', 'SELECT']);

function shouldSkip(text) {
    if (SKIP_WORDS.has(text.toLowerCase())) return true;
    if (/^\d[\d\/\-\.:, ]*(\s*(kB|MB|GB|KB|pm|am))?$/i.test(text)) return true;
    if (/^[a-z]+-[a-z][-a-z]*$/.test(text)) return true;
    if (text.length < 5 && !text.includes(' ')) return true;
    if (text.charAt(0) === '{' || text.charAt(0) === '[') return true;  // JSON
    return false;
}


function isInEditableArea(el) {
    let current = el;
    for (let i = 0; i < 10 && current; i++) {
        if (current.isContentEditable) return true;
        if (current.getAttribute && current.getAttribute('role') === 'textbox') return true;
        current = current.parentElement;
    }
    return false;
}

function isInChatPanel(el) {
    const mainPanel = document.getElementById('main');
    if (mainPanel && mainPanel.contains(el)) return true;
    return false;
}

function findTextElements(root) {
    const elements = [];
    
    const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: function(node) {
                const text = node.textContent.trim();
                if (text.length < MIN_TEXT_LENGTH) return NodeFilter.FILTER_REJECT;
                
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                if (SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
                
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );

    const seen = new Set();
    while (walker.nextNode()) {
        const el = walker.currentNode.parentElement;
        if (!el) continue;
        if (seen.has(el)) continue;
        if (el.hasAttribute(PROCESSED_ATTR)) continue;
        if (isInEditableArea(el)) continue;
        
        const text = el.textContent.trim();
        if (text.length > 500) continue;   // Skip huge containers
        if (shouldSkip(text)) {
            el.setAttribute(PROCESSED_ATTR, 'skip');
            continue;
        }
        
        seen.add(el);
        elements.push(el);
    }

    return elements;
}


function isInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.bottom > 0 &&
        rect.right > 0 &&
        rect.top < window.innerHeight &&
        rect.left < window.innerWidth
    );
}

const checkedSrcMap = new WeakMap();

function findImages(root) {
    if (!root.querySelectorAll) return [];
    
    const isImgSearch = isImageSearchPage();
    const images = [];
  
    const selector = isImgSearch ? 'img' : 'img:not([data-safevision])';
    const allImgs = root.querySelectorAll(selector);
    
    for (const img of allImgs) {
        // Skip images NOT visible on screen
        if (!isInViewport(img)) continue;
        
        // Get dimensions
        const w = img.naturalWidth || img.width || img.offsetWidth || 0;
        const h = img.naturalHeight || img.height || img.offsetHeight || 0;
        
        // Skip tiny images (icons, 1x1 placeholders)
        if (w < 40 || h < 40) continue;
        
        const url = img.src;
        if (!url) continue;
        
        // Skip placeholder GIFs and SVGs
        if (url.startsWith('data:image/gif')) continue;
        if (url.startsWith('data:') && url.includes('svg')) continue;
        
        // Skip emoji/icon images
        const alt = (img.alt || '').toLowerCase();
        const cls = (img.className || '').toLowerCase();
        if (alt.includes('emoji') || alt.includes('icon') || cls.includes('icon')) continue;
        
        // On image search: skip if we already checked THIS exact src
        if (isImgSearch && img.hasAttribute(PROCESSED_ATTR)) {
            const lastCheckedSrc = checkedSrcMap.get(img);
            if (lastCheckedSrc === url) continue;  // Same src, already checked → skip
            // src changed → remove old mark, will be re-scanned
            img.removeAttribute(PROCESSED_ATTR);
            img.classList.remove('safevision-blur-img');
        }
        
        // Track which src we're about to check
        checkedSrcMap.set(img, url);
        images.push(img);
    }
    
    return images;
}

// ========================================
// CHECK ONE ELEMENT (with retry)
// ========================================
async function checkAndBlur(el) {
    const text = el.textContent.trim();
    if (!text || text.length < MIN_TEXT_LENGTH) return;

    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            const result = await chrome.runtime.sendMessage({
                action: "checkText",
                text: text
            });

            if (result && result.status === 'unsafe') {
                console.log('%c🚫 SafeVision BLUR: ' + text.substring(0, 60),
                    'color: red; font-weight: bold; font-size: 13px;');
                el.classList.add('safevision-blur');
                el.title = 'SafeVision: Toxic Content Blocked';
                el.setAttribute(PROCESSED_ATTR, 'blocked');
            } else {
                el.setAttribute(PROCESSED_ATTR, 'safe');
            }
            return; // Success, stop retrying
        } catch (e) {
            if (attempt === 0) {
                await sleep(500); // Wait and retry once
            }
            // On second failure, just mark and move on
            el.setAttribute(PROCESSED_ATTR, 'error');
        }
    }
}

// ========================================
// CONVERT IMAGE TO BASE64 — resized to 224x224
// ViT model input is 224x224 anyway — resize here cuts payload 5x.
// Smaller payload = faster canvas encode + faster network = faster blur.
// ========================================
function imgToBase64(img) {
    return new Promise((resolve) => {
        try {
            const canvas = document.createElement('canvas');
            canvas.width  = 224;  // ViT input size
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 224, 224);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
            resolve(dataUrl);
        } catch (e) {
            resolve(null);
        }
    });
}

// ========================================
// IMAGE BATCH QUEUE
// Collect images for 300ms then send all at once to /predict_images_batch
// GPU processes the whole batch in one forward pass
// ========================================
let imgBatchQueue = [];   // { id, img, imageData, resolve }
let imgBatchTimer = null;

function flushImageBatch() {
    imgBatchTimer = null;
    if (imgBatchQueue.length === 0) return;

    const batch = imgBatchQueue.splice(0);
    const payload = batch.map(item => ({ id: item.id, imageData: item.imageData }));

    console.log(`[SafeVision] Sending batch of ${batch.length} images`);

    chrome.runtime.sendMessage({ action: "checkImageBatch", images: payload })
        .then(response => {
            if (!response || !response.results) {
                // Fallback: mark all as error
                batch.forEach(item => item.resolve('error'));
                return;
            }
            const resultMap = {};
            response.results.forEach(r => { resultMap[r.id] = r.status; });
            batch.forEach(item => item.resolve(resultMap[item.id] || 'safe'));
        })
        .catch(() => {
            batch.forEach(item => item.resolve('error'));
        });
}

// ========================================
// CHECK ONE IMAGE — queues into batch
// ========================================
async function checkAndBlurImage(img) {
    const url = img.src;
    if (!url) return;

    // Convert to base64
    let imageData = null;
    if (url.startsWith('data:')) {
        imageData = url;
    } else {
        imageData = await imgToBase64(img);
        if (!imageData) {
            img.setAttribute(PROCESSED_ATTR, 'error');
            return;
        }
    }

    // Generate a unique ID for this image in the batch
    const id = Math.random().toString(36).slice(2, 10);

    // Add to batch and return a promise that resolves when the batch comes back
    const status = await new Promise(resolve => {
        imgBatchQueue.push({ id, img, imageData, resolve });

        // Start/reset the flush timer
        if (imgBatchTimer) clearTimeout(imgBatchTimer);
        imgBatchTimer = setTimeout(flushImageBatch, IMG_BATCH_WINDOW_MS);
    });

    // Apply result
    if (status === 'unsafe') {
        console.log('%c[SafeVision] BLOCKED IMAGE: ' + url.substring(0, 80),
            'color: red; font-weight: bold;');
        img.classList.add('safevision-blur-img');
        img.title = 'SafeVision: NSFW Image Blocked';
        img.setAttribute(PROCESSED_ATTR, 'blocked');
    } else {
        img.setAttribute(PROCESSED_ATTR, status === 'error' ? 'error' : 'safe');
    }
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// ========================================
// PROCESS ELEMENTS IN BATCHES
// Simple and reliable — no complex queue
// ========================================
const processingLocks = new Map();  // lockKey -> timestamp
const LOCK_TIMEOUT_MS = 10000;      // Auto-release stuck locks after 10s

function isLockStale(lockKey) {
    if (!processingLocks.has(lockKey)) return false;
    return (Date.now() - processingLocks.get(lockKey)) > LOCK_TIMEOUT_MS;
}

async function processBatch(elements, handler, lockKey) {
    if (elements.length === 0) return;
    
    // If lock exists but is stale (>10s), force release it
    if (processingLocks.has(lockKey)) {
        if (isLockStale(lockKey)) {
            console.log(`[SafeVision] Lock '${lockKey}' was stuck for >10s — releasing`);
            processingLocks.delete(lockKey);
        } else {
            return; // Lock is fresh, skip
        }
    }
    processingLocks.set(lockKey, Date.now());

    try {
        for (let i = 0; i < elements.length; i += BATCH_SIZE) {
            const batch = elements.slice(i, i + BATCH_SIZE);
            
            // Process batch in parallel
            await Promise.all(batch.map(el => handler(el)));
            
            // Small delay between batches to keep page responsive
            if (i + BATCH_SIZE < elements.length) {
                await sleep(BATCH_DELAY);
            }
        }
    } catch (e) {
        console.error('SafeVision batch error:', e);
    }

    processingLocks.delete(lockKey);
}

// ========================================
// MAIN SCAN — smart priority based on page type
// ========================================
function isImageSearchPage() {
    const url = window.location.href;
    return (url.includes('tbm=isch') || url.includes('udm=2') ||
            url.includes('/images?') || url.includes('bing.com/images'));
}

/**
 * Find the MAIN (largest) image in the Google Images detail panel.
 * When you click a thumbnail, Google opens a panel with the full-size image.
 * We want to scan THAT image, not the tiny recommended thumbnails.
 */
function findMainPanelImage() {
    if (!document.querySelectorAll) return null;

    // Google Images panel selectors (the full-size image shown when you click)
    const panelSelectors = [
        'div[data-ved] img[src*="encrypted"]',   // encrypted image in panel
        '.islsp img',                             // image in side panel
        '.OXftbe img',                           // alt panel class
        '[data-id] img.iPVvYb',                  // large image variant
        'c-wiz img[src]:not([src*=".gif"]):not([src*="data:image/gif"])',
    ];

    let bestImg = null;
    let bestArea = 0;

    // Also just find the largest img on page (by rendered size)
    const allImgs = document.querySelectorAll('img[src]:not([data-safevision])');
    for (const img of allImgs) {
        // Skip tiny icons/thumbnails — panel image is always large
        const w = img.naturalWidth || img.offsetWidth || 0;
        const h = img.naturalHeight || img.offsetHeight || 0;
        const area = w * h;
        if (area > bestArea && w > 200 && h > 200) {
            bestArea = area;
            bestImg = img;
        }
    }

    return bestImg;
}

const IS_CHAT_SITE = window.location.hostname.includes('whatsapp.com') ||
                     window.location.hostname.includes('instagram.com') ||
                     window.location.hostname.includes('messenger.com');

// Routes to the correct handler based on element type
async function checkAndBlurAny(el) {
    if (el.tagName === 'IMG') {
        return checkAndBlurImage(el);
    } else {
        return checkAndBlur(el);
    }
}

// Sort elements by their position in the DOM (top→bottom)
function sortByDomOrder(elements) {
    return elements.sort((a, b) => {
        const pos = a.compareDocumentPosition(b);
        if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
        if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
        return 0;
    });
}

async function scan() {
    const chatPanel = document.getElementById('main');
    const isImgSearch = isImageSearchPage();
    
    // ==== IMAGE SEARCH: scan grid thumbnails only ====
    if (isImgSearch) {
        // Always clear the lock — image search needs constant re-scanning
        processingLocks.delete('imgSearch');
        const images = findImages(document.body);
        if (images.length > 0) {
            console.log(`[SafeVision] IMAGE SEARCH GRID: ${images.length} visible images`);
            await processBatch(images, checkAndBlurImage, 'imgSearch');
        }
        return;
    }
    
    // ==== CHAT SITES: Chat panel first (newest→oldest), sidebar later ====
    if (chatPanel) {
        const chatText = findTextElements(chatPanel);
        const chatImages = findImages(chatPanel);
        
        // Merge text + images into one array, sorted by DOM position
        const chatAll = sortByDomOrder([...chatText, ...chatImages]);
        
        // For chat sites, reverse to scan newest messages first
        if (IS_CHAT_SITE) {
            chatAll.reverse();
        }
        
        if (chatAll.length > 0) {
            const orderStr = IS_CHAT_SITE ? 'newest→oldest' : 'top→bottom';
            console.log(`[SafeVision] ⚡ MAIN (${orderStr}): ${chatAll.length} elements (${chatText.length} text, ${chatImages.length} images)`);
            await processBatch(chatAll, checkAndBlurAny, 'chat');
        }
    }
    
    // ==== REST: Sidebar/UI ====
    const restText = findTextElements(document.body);
    const restImages = findImages(document.body);
    
    // Merge text + images into one DOM-ordered array
    const restAll = sortByDomOrder([...restText, ...restImages]);
    
    if (restAll.length > 0) {
        console.log(`[SafeVision] 📋 Sidebar/UI: ${restAll.length} elements (${restText.length} text, ${restImages.length} images)`);
        processBatch(restAll, checkAndBlurAny, 'side');
    }
}

// ========================================
// MUTATION OBSERVER — smart chat detection
// ========================================
let scanTimer = null;
let lastChatName = null;

function debouncedScan() {
    if (scanTimer) clearTimeout(scanTimer);
    scanTimer = setTimeout(scan, 300);
}

// Force rescan chat panel (used on chat switch)
function forceRescanChat() {
    const chatPanel = document.getElementById('main');
    if (!chatPanel) return;
    
    // Clear all processed marks inside the chat panel
    const processed = chatPanel.querySelectorAll(`[${PROCESSED_ATTR}]`);
    processed.forEach(el => el.removeAttribute(PROCESSED_ATTR));
    
    // Clear all locks so scan() can run
    processingLocks.clear();
    
    console.log('[SafeVision] 🔄 Chat switch detected — rescanning chat...');
    scan();
    attachScrollListeners();
}

// Detect chat switches by checking the chat header name
function getChatName() {
    const header = document.querySelector('#main header');
    if (!header) return null;
    // Get the first significant text (the contact/group name)
    const nameEl = header.querySelector('span[dir="auto"], span[title]');
    return nameEl ? nameEl.textContent.trim() : header.textContent.trim().substring(0, 30);
}

const observer = new MutationObserver((mutations) => {
    let hasNewInChat = false;
    let hasNewAnywhere = false;
    
    for (const mutation of mutations) {
        // Handle src changes — Google Images reuses <img> elements with new src
        // Without this, images get marked data-safevision once and never re-checked
        if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
            const img = mutation.target;
            if (img.tagName === 'IMG' && img.hasAttribute(PROCESSED_ATTR)) {
                img.removeAttribute(PROCESSED_ATTR);           // Reset → will be re-scanned
                img.classList.remove('safevision-blur-img');    // Remove old blur
            }
            hasNewAnywhere = true;
            continue;
        }
        
        if (mutation.type !== 'childList' || mutation.addedNodes.length === 0) continue;
        hasNewAnywhere = true;
        
        // Check if changes are inside the chat panel
        const mainPanel = document.getElementById('main');
        if (mainPanel && mainPanel.contains(mutation.target)) {
            hasNewInChat = true;
        }
    }
    
    if (!hasNewAnywhere) return;
    
    // Check for chat switch
    const currentChat = getChatName();
    if (currentChat && currentChat !== lastChatName) {
        lastChatName = currentChat;
        forceRescanChat();
        return;
    }
    
    // New content in chat (user sent/received a message)
    if (hasNewInChat) {
        processingLocks.delete('chat');
        debouncedScan();
    } else {
        debouncedScan();
    }
});

// ========================================
// PERIODIC RE-SCAN (gentle — only for new content)
// ========================================
if (isImageSearchPage()) {
    // Image search: scan every 3s with debug logging
    setInterval(() => {
        const allImgs = document.querySelectorAll('img');
        const inViewport = [...allImgs].filter(img => isInViewport(img) && (img.naturalWidth || img.offsetWidth || 0) > 40);
        const unprocessed = inViewport.filter(img => {
            if (!img.hasAttribute(PROCESSED_ATTR)) return true;
            const lastSrc = checkedSrcMap.get(img);
            return lastSrc !== img.src;  // src changed
        });
        console.log(`[SafeVision] PERIODIC: ${allImgs.length} total imgs, ${inViewport.length} in viewport, ${unprocessed.length} need scan`);
        processingLocks.delete('imgSearch');
        scan();
    }, 3000);
} else {
    // Other sites: scan every 8s
    setInterval(() => {
        for (const [key, time] of processingLocks) {
            if (Date.now() - time > LOCK_TIMEOUT_MS) {
                processingLocks.delete(key);
            }
        }
        scan();
    }, 8000);
}

// ========================================
// SCROLL LISTENER — scan newly visible content
// ========================================
let imgSearchScrollTimer = null;

function attachScrollListeners() {
    // Google Images: scan after every scroll stop
    if (isImageSearchPage()) {
        if (!document.body.hasAttribute('data-sv-scroll')) {
            document.body.setAttribute('data-sv-scroll', 'true');
            window.addEventListener('scroll', () => {
                // Wait 500ms after scroll stops, then scan
                if (imgSearchScrollTimer) clearTimeout(imgSearchScrollTimer);
                imgSearchScrollTimer = setTimeout(() => {
                    processingLocks.delete('imgSearch');
                    scan();
                }, 500);
            }, { passive: true });
        }
        return;
    }
    
    const chatPanel = document.getElementById('main');
    if (chatPanel) {
        const containers = chatPanel.querySelectorAll('[tabindex="-1"]');
        containers.forEach(c => {
            if (!c.hasAttribute('data-sv-scroll')) {
                c.setAttribute('data-sv-scroll', 'true');
                c.addEventListener('scroll', debouncedScan, { passive: true });
            }
        });
    }
}

// ========================================
// CLICK LISTENER — Google Images
// On click: immediately scan the MAIN full-size image only
// Skip tiny recommended thumbnails in the panel
// ========================================
document.addEventListener('click', () => {
    if (!isImageSearchPage()) return;

    // Wait for Google to load the panel image (it lazy-loads after click)
    setTimeout(async () => {
        const mainImg = findMainPanelImage();
        if (mainImg && !mainImg.hasAttribute(PROCESSED_ATTR)) {
            const w = mainImg.naturalWidth || mainImg.offsetWidth || 0;
            const h = mainImg.naturalHeight || mainImg.offsetHeight || 0;
            console.log(`[SafeVision] PANEL IMAGE: ${w}x${h} — scanning now`);
            await checkAndBlurImage(mainImg);
        }

        // Also do a regular grid scan (picks up any new thumbnails)
        processingLocks.delete('imgSearch');
        scan();
    }, 600); // 600ms lets Google finish loading the panel
}, { passive: true });


// ========================================
// START
// ========================================
function start() {
    console.log("[SafeVision] Starting...");
    // Watch childList AND attributes (catches Google changing img src on lazy-load)
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    lastChatName = getChatName();
    scan();
    attachScrollListeners();
    
    // Re-attach scroll listeners periodically
    setInterval(attachScrollListeners, 5000);
}

if (document.body) {
    start();
} else {
    document.addEventListener('DOMContentLoaded', start);
}