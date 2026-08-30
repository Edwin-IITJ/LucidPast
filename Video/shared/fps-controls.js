/**
 * fps-controls.js — Shared FPS camera controls for LucidPast viewers.
 *
 * Ported from the image pipeline's viewer.js so both the image and video
 * viewers use identical navigation. No dependency on GaussianSplats3D or
 * any specific renderer — only needs a Three.js camera and a DOM element.
 *
 * Controls:
 *   Mouse drag:  Look around (yaw / pitch)
 *   W / S:       Move forward / backward
 *   A / D:       Strafe left / right
 *   Q / E:       Move down / up
 *   Scroll:      Adjust movement speed (Unreal Engine style)
 *
 * Usage (ES module):
 *   import { FPSControls } from './fps-controls.js';
 *   const controls = new FPSControls(camera, containerElement);
 *   // in your animation loop:
 *   controls.update();
 *   // to clean up:
 *   controls.dispose();
 */

import * as THREE from "three";

export class FPSControls {
  constructor(camera, domElement, options = {}) {
    this.camera = camera;
    this.domElement = domElement;

    // Configurable
    this.moveSpeed = options.moveSpeed ?? 0.05;
    this.lookSpeed = options.lookSpeed ?? 0.003;
    this.minSpeed = options.minSpeed ?? 0.001;
    this.maxSpeed = options.maxSpeed ?? 5.0;
    this.speedMultiplier = options.speedMultiplier ?? 1.2;

    // Internal state
    this.keys = { w: false, a: false, s: false, d: false, q: false, e: false };
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };

    // Euler for rotation (YXZ order — yaw then pitch)
    this.euler = new THREE.Euler(0, 0, 0, "YXZ");
    this.euler.setFromQuaternion(this.camera.quaternion);

    // Bind handlers so we can remove them later
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    this._onContextMenu = (e) => e.preventDefault();

    this._attach();
  }

  _attach() {
    this.domElement.addEventListener("mousedown", this._onMouseDown);
    document.addEventListener("mousemove", this._onMouseMove);
    document.addEventListener("mouseup", this._onMouseUp);
    this.domElement.addEventListener("wheel", this._onWheel, { passive: false });
    this.domElement.addEventListener("contextmenu", this._onContextMenu);
    document.addEventListener("keydown", this._onKeyDown);
    document.addEventListener("keyup", this._onKeyUp);
  }

  _onMouseDown() {
    this.isDragging = true;
  }

  _onMouseMove(e) {
    if (this.isDragging) {
      const deltaX = e.movementX || (e.clientX - this.previousMousePosition.x);
      const deltaY = e.movementY || (e.clientY - this.previousMousePosition.y);

      this.euler.y -= deltaX * this.lookSpeed;
      this.euler.x -= deltaY * this.lookSpeed;

      // Clamp pitch to ±90°
      this.euler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.euler.x));

      this.camera.quaternion.setFromEuler(this.euler);
    }
    this.previousMousePosition = { x: e.clientX, y: e.clientY };
  }

  _onMouseUp() {
    this.isDragging = false;
  }

  _onWheel(e) {
    e.preventDefault();
    if (e.deltaY < 0) {
      this.moveSpeed = Math.min(this.moveSpeed * this.speedMultiplier, this.maxSpeed);
    } else if (e.deltaY > 0) {
      this.moveSpeed = Math.max(this.moveSpeed / this.speedMultiplier, this.minSpeed);
    }
  }

  _onKeyDown(e) {
    const key = e.key.toLowerCase();
    if (key in this.keys) this.keys[key] = true;
  }

  _onKeyUp(e) {
    const key = e.key.toLowerCase();
    if (key in this.keys) this.keys[key] = false;
  }

  /** Call this every frame in your animation loop. */
  update() {
    if (!this.camera) return;

    const direction = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);

    this.camera.getWorldDirection(direction);
    direction.y = 0;
    direction.normalize();

    right.crossVectors(this.camera.up, direction).normalize();

    if (this.keys.w) this.camera.position.addScaledVector(direction, this.moveSpeed);
    if (this.keys.s) this.camera.position.addScaledVector(direction, -this.moveSpeed);
    if (this.keys.a) this.camera.position.addScaledVector(right, this.moveSpeed);
    if (this.keys.d) this.camera.position.addScaledVector(right, -this.moveSpeed);
    if (this.keys.e) this.camera.position.addScaledVector(up, this.moveSpeed);
    if (this.keys.q) this.camera.position.addScaledVector(up, -this.moveSpeed);
  }

  dispose() {
    this.domElement.removeEventListener("mousedown", this._onMouseDown);
    document.removeEventListener("mousemove", this._onMouseMove);
    document.removeEventListener("mouseup", this._onMouseUp);
    this.domElement.removeEventListener("wheel", this._onWheel);
    this.domElement.removeEventListener("contextmenu", this._onContextMenu);
    document.removeEventListener("keydown", this._onKeyDown);
    document.removeEventListener("keyup", this._onKeyUp);
  }
}
