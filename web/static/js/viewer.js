// viewer.js - Handles Three.js and Gaussian Splats

class SplatViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.viewer = null;
        this.currentSceneName = null;
    }

    async init() {
        // Clear container if needed
        this.container.innerHTML = '';
        
        // Initialize viewer without built-in controls
        this.viewer = new GaussianSplats3D.Viewer({
            selfDrivenMode: true,
            useBuiltInControls: false,
            rootElement: this.container,
            cameraUp: [0, -1, -0.6], // SHARP y-axis might need adjustment, this is a common preset
            initialCameraPosition: [0, 0, 5],
            initialCameraLookAt: [0, 0, 0],
            backgroundColor: '#000000',
            webXRMode: GaussianSplats3D.WebXRMode.VR, // enable VR capabilities
            sharedMemoryForWorkers: true // Re-enable for full fidelity now that headers are fixed
        });

        await this.viewer.init();
        
        // Expose underlying Three.js objects for xr.js
        this.scene = this.viewer.splatMesh.scene || this.viewer.scene;
        this.camera = this.viewer.camera;
        this.renderer = this.viewer.renderer;
        
        this.setupCustomControls();
    }

    setupCustomControls() {
        // --- FPS Camera State ---
        this.keys = { w: false, a: false, s: false, d: false, q: false, e: false };
        this.isDragging = false;
        this.previousMousePosition = { x: 0, y: 0 };
        this.moveSpeed = 0.05;
        this.lookSpeed = 0.003;
        
        // --- Euler for Rotation ---
        this.euler = new THREE.Euler(0, 0, 0, 'YXZ');
        this.euler.setFromQuaternion(this.camera.quaternion);

        // --- Mouse Events ---
        this.container.addEventListener('mousedown', (e) => {
            this.isDragging = true;
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaX = e.movementX || (e.clientX - this.previousMousePosition.x);
                const deltaY = e.movementY || (e.clientY - this.previousMousePosition.y);
                
                this.euler.y -= deltaX * this.lookSpeed;
                this.euler.x -= deltaY * this.lookSpeed;
                
                // Clamp pitch
                this.euler.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.euler.x));
                
                this.camera.quaternion.setFromEuler(this.euler);
            }
            this.previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        document.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        // --- Scroll Wheel for Speed Adjustment (Unreal Engine style) ---
        this.container.addEventListener('wheel', (e) => {
            // Only adjust speed if we are holding a mouse button (like right-click dragging)
            // or we can allow it anytime. Let's allow it anytime the mouse is over the viewer.
            e.preventDefault();
            const speedMultiplier = 1.2;
            if (e.deltaY < 0) {
                // Scroll up -> Increase speed
                this.moveSpeed = Math.min(this.moveSpeed * speedMultiplier, 5.0);
            } else if (e.deltaY > 0) {
                // Scroll down -> Decrease speed
                this.moveSpeed = Math.max(this.moveSpeed / speedMultiplier, 0.001);
            }
            console.log(`Camera Speed: ${this.moveSpeed.toFixed(4)}`);
        }, { passive: false });

        // --- Keyboard Events ---
        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() in this.keys) this.keys[e.key.toLowerCase()] = true;
        });

        document.addEventListener('keyup', (e) => {
            if (e.key.toLowerCase() in this.keys) this.keys[e.key.toLowerCase()] = false;
        });

        // --- Movement Loop ---
        const updateMovement = () => {
            requestAnimationFrame(updateMovement);
            if (!this.camera) return;

            const direction = new THREE.Vector3();
            const right = new THREE.Vector3();
            const up = new THREE.Vector3(0, 1, 0); // Global up vector for Q/E

            this.camera.getWorldDirection(direction);
            direction.y = 0; // Move horizontally
            direction.normalize();

            right.crossVectors(this.camera.up, direction).normalize();

            if (this.keys.w) this.camera.position.addScaledVector(direction, this.moveSpeed);
            if (this.keys.s) this.camera.position.addScaledVector(direction, -this.moveSpeed);
            if (this.keys.a) this.camera.position.addScaledVector(right, this.moveSpeed);
            if (this.keys.d) this.camera.position.addScaledVector(right, -this.moveSpeed);
            if (this.keys.e) this.camera.position.addScaledVector(up, this.moveSpeed);
            if (this.keys.q) this.camera.position.addScaledVector(up, -this.moveSpeed);
        };

        updateMovement();
    }

    async loadScene(sceneName, fileName) {
        if (!this.viewer) {
            await this.init();
        }

        const url = `/api/splat/${sceneName}/${fileName}`;
        console.log(`Loading splat from: ${url}`);
        
        try {
            // Remove existing splats if any
            if (this.viewer.splatMesh) {
                // Simplified reset, in a robust app we'd properly dispose meshes
                this.viewer.dispose();
                await this.init();
            }

            await this.viewer.addSplatScene(url, {
                splatAlphaCrop: 0.1,
                showLoadingUI: false // Fix LoadingSpinner removeChild bug
            });
            
            this.viewer.start();
            this.currentSceneName = sceneName;
            return true;
        } catch (err) {
            console.error("Failed to load splat scene:", err);
            return false;
        }
    }
}

// Global instance
window.lucidViewer = new SplatViewer('viewer-container');
