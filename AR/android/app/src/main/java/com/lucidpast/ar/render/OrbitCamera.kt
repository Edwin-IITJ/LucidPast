package com.lucidpast.ar.render

import android.opengl.Matrix
import kotlin.math.cos
import kotlin.math.sin

class OrbitCamera {

    var distance = 0.9f
        private set
    var yaw = 0.6f
        private set
    var pitch = 0.3f
        private set

    val viewMatrix = FloatArray(16)

    init {
        update()
    }

    fun update() {
        val cp = cos(pitch)
        val ex = distance * cp * sin(yaw)
        val ey = distance * sin(pitch)
        val ez = distance * cp * cos(yaw)
        Matrix.setLookAtM(viewMatrix, 0, ex, ey, ez, 0f, 0f, 0f, 0f, 1f, 0f)
    }

    fun reset(captureDistance: Float) {
        yaw = 0f
        pitch = 0f
        distance = captureDistance.coerceIn(0.15f, 4.0f)
        update()
    }

    fun drag(dx: Float, dy: Float) {
        yaw -= dx * 0.006f
        pitch = (pitch + dy * 0.006f).coerceIn(-1.35f, 1.35f)
        update()
    }

    fun zoom(factor: Float) {
        distance = (distance * factor).coerceIn(0.15f, 4.0f)
        update()
    }
}
