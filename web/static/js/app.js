// app.js - Main application state machine

function initApp() {
    // --- UI Elements ---
    const panelUpload = document.getElementById('panel-upload');
    const panelProcessing = document.getElementById('panel-processing');
    const panelViewer = document.getElementById('panel-viewer');
    
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const terminalOutput = document.getElementById('terminal-output');
    const currentSceneNameLabel = document.getElementById('current-scene-name');

    
    const btnSkipToGallery = document.getElementById('btn-skip-to-gallery');
    const btnBack = document.getElementById('btn-back');
    const galleryList = document.getElementById('gallery-list');
    
    // --- State ---
    let currentJobId = null;
    let pollInterval = null;
    let lastLogIdx = 0;
    
    // Initialize XR Manager
    const xrManager = new XRManager(window.lucidViewer);
    
    // --- Navigation Functions ---
    function showPanel(panel) {
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        panel.classList.add('active');
        
        if (panel === panelViewer) {
            xrManager.init();
            loadGallery();
        }
    }
    
    // --- Upload Handlers ---
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    btnSkipToGallery.addEventListener('click', () => {
        showPanel(panelViewer);
        // Viewer is initially empty until a gallery item is clicked
        currentSceneNameLabel.textContent = "Select a memory from the archive";
    });
    
    btnBack.addEventListener('click', () => {
        showPanel(panelUpload);
    });
    
    // --- API Interactions ---
    async function handleFileUpload(file) {
        showPanel(panelProcessing);
        terminalOutput.innerHTML = '';
        appendLog(`[INFO] Preparing to upload ${file.name}...`);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok && data.job_id) {
                currentJobId = data.job_id;
                lastLogIdx = 0;
                startPolling();
            } else {
                appendLog(`[ERROR] ${data.error || 'Upload failed'}`);
                setTimeout(() => showPanel(panelUpload), 3000);
            }
        } catch (err) {
            appendLog(`[ERROR] Network error: ${err.message}`);
            setTimeout(() => showPanel(panelUpload), 3000);
        }
    }
    
    function appendLog(msg) {
        const line = document.createElement('div');
        line.className = 'terminal-line';
        line.textContent = msg;
        terminalOutput.appendChild(line);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }
    
    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        
        pollInterval = setInterval(async () => {
            if (!currentJobId) return;
            
            try {
                const res = await fetch(`/api/status/${currentJobId}?last_log_idx=${lastLogIdx}`);
                const data = await res.json();
                
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => appendLog(log));
                }
                
                lastLogIdx = data.next_log_idx;
                
                if (data.status === 'done') {
                    clearInterval(pollInterval);
                    appendLog(`[INFO] Rendering scene in browser...`);
                    setTimeout(() => {
                        showPanel(panelViewer);
                        // Fetch the file name (usually <name>.ply, we'll just fetch gallery list and match)
                        loadGalleryAndSelect(data.image_name);
                    }, 1000);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    appendLog(`[ERROR] Processing failed.`);
                    setTimeout(() => showPanel(panelUpload), 3000);
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 1000); // Poll every second
    }
    
    // --- Gallery Management ---
    async function loadGallery() {
        try {
            const res = await fetch('/api/splats');
            const data = await res.json();
            
            galleryList.innerHTML = '';
            
            if (data.splats && data.splats.length > 0) {
                data.splats.forEach(splat => {
                    const item = document.createElement('div');
                    item.className = 'gallery-item';
                    item.textContent = splat.name;
                    item.dataset.name = splat.name;
                    item.dataset.file = splat.file;
                    
                    item.addEventListener('click', () => {
                        // Update UI
                        document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('active'));
                        item.classList.add('active');
                        currentSceneNameLabel.textContent = splat.name;
                        
                        // Load into Viewer
                        window.lucidViewer.loadScene(splat.name, splat.file);
                    });
                    
                    galleryList.appendChild(item);
                });
            } else {
                galleryList.innerHTML = '<div style="opacity: 0.5; font-size: 0.8rem;">No memories archived yet.</div>';
            }
        } catch (err) {
            console.error("Failed to load gallery:", err);
        }
    }
    
    async function loadGalleryAndSelect(sceneName) {
        await loadGallery();
        const items = document.querySelectorAll('.gallery-item');
        for (const item of items) {
            if (item.dataset.name === sceneName) {
                item.click();
                return;
            }
        }
        // If not found by name, just click the first one
        if (items.length > 0) items[0].click();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
