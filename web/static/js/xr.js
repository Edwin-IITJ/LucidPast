// xr.js - Handles WebXR integration

class XRManager {
    constructor(viewerInstance) {
        this.viewer = viewerInstance;
        this.xrButtonContainer = document.getElementById('xr-controls');
        this.vrSupported = false;
    }

    init() {
        if ('xr' in navigator) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                this.vrSupported = supported;
                if (supported) {
                    this.createVRButton();
                } else {
                    console.warn("WebXR immersive-vr not supported on this device/browser.");
                    this.createVRButton(false);
                }
            });
        } else {
            console.warn("WebXR API not found.");
            this.createVRButton(false);
        }
    }

    createVRButton(supported = true) {
        // Remove old button if exists
        const oldBtn = document.getElementById('btn-enter-vr');
        if (oldBtn) oldBtn.remove();

        const btn = document.createElement('button');
        btn.id = 'btn-enter-vr';
        
        if (supported) {
            btn.className = 'btn btn-primary';
            btn.textContent = 'Enter VR (Quest)';
            btn.onclick = () => this.enterVR();
        } else {
            btn.className = 'btn';
            btn.textContent = 'VR Not Supported';
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
            btn.title = "Use a WebXR compatible browser like Oculus Browser on Quest";
            btn.disabled = true;
        }

        // Insert before the Back button
        this.xrButtonContainer.insertBefore(btn, this.xrButtonContainer.firstChild);
    }

    enterVR() {
        if (!this.viewer.renderer) {
            console.error("Renderer not ready");
            return;
        }
        
        const renderer = this.viewer.renderer;
        renderer.xr.enabled = true;
        
        // Use Three.js VRButton module logic essentially
        const sessionInit = { optionalFeatures: ['local-floor', 'bounded-floor'] };
        
        navigator.xr.requestSession('immersive-vr', sessionInit).then((session) => {
            session.addEventListener('end', () => this.onSessionEnded());
            renderer.xr.setSession(session);
            
            // Adjust camera for VR void
            this.setupVREnvironment();
            
        }).catch(err => {
            console.error("Failed to enter VR:", err);
            alert("Could not start VR session. Please ensure your headset is connected.");
        });
    }

    setupVREnvironment() {
        // Ensure pure black void
        if (this.viewer.scene) {
            this.viewer.scene.background = new THREE.Color(0x000000);
            this.viewer.scene.fog = null; // Eulogy style: pure void
        }
    }

    onSessionEnded() {
        console.log("VR Session ended.");
        // Renderer xr session is automatically cleaned up by Three.js
    }
}

// Global initialization happens in app.js
