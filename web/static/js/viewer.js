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
        
        // Initialize viewer
        this.viewer = new GaussianSplats3D.Viewer({
            selfDrivenMode: true,
            useBuiltInControls: true,
            rootElement: this.container,
            cameraUp: [0, -1, -0.6], // SHARP y-axis might need adjustment, this is a common preset
            initialCameraPosition: [0, 0, 5],
            initialCameraLookAt: [0, 0, 0],
            backgroundColor: '#000000',
            webXRMode: GaussianSplats3D.WebXRMode.VR, // enable VR capabilities
            sharedMemoryForWorkers: false // Fix SharedArrayBuffer cross-origin issue
        });

        await this.viewer.init();
        
        // Expose underlying Three.js objects for xr.js
        this.scene = this.viewer.splatMesh.scene || this.viewer.scene;
        this.camera = this.viewer.camera;
        this.renderer = this.viewer.renderer;
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
