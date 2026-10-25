package com.lucidpast.ar.render

import android.opengl.GLES31
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.tan

class SplatRenderer : GLSurfaceView.Renderer {

    companion object {
        const val DEPTH_BUCKETS = 65536
        const val WORKGROUP = 256
        const val SCAN_BLOCKS = DEPTH_BUCKETS / WORKGROUP
        private const val FOV_Y_DEG = 50f
        private const val RECORD_BYTES = 32
        private const val PREPARED_BYTES = 64
        private const val Z_NEAR = 0.02f
        const val VIDEO_FIT_DIST = 1.2f
    }

    private object B {
        const val RECORDS = 0
        const val COUNTS = 1
        const val CURSOR = 2
        const val SCANNED = 3
        const val BLOCKS = 4
        const val ORDER = 5
        const val PREPARED = 6
    }

    private class PendingLoad(
        val records: ByteBuffer,
        val count: Int,
        val model: FloatArray,
        val captureDist: Float,
        val resetCamera: Boolean,
    )

    private val buffers = IntArray(7)
    private var depthProg = 0
    private var scanBlocksProg = 0
    private var scanSumsProg = 0
    private var scanOffsetsProg = 0
    private var scatterProg = 0
    private var prepareProg = 0
    private var drawProg = 0

    var camera = OrbitCamera()
        private set
    var viewDirty = true
    var framesDrawn = 0
        private set
    var lastSortMs = 0L
        private set
    var lastSortCount = 0
        private set

    private var splatCount = 0
    private val modelMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private var viewportW = 1
    private var viewportH = 1
    private var focal = 1f
    private val zeroBuf =
        ByteBuffer.allocateDirect(DEPTH_BUCKETS * 4).order(ByteOrder.nativeOrder())
    private var surfaceReady = false
    private var pendingLoad: PendingLoad? = null

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        depthProg = buildComputeProgram(Shaders.DEPTH_COUNT)
        scanBlocksProg = buildComputeProgram(Shaders.SCAN_BLOCKS)
        scanSumsProg = buildComputeProgram(Shaders.SCAN_SUMS)
        scanOffsetsProg = buildComputeProgram(Shaders.SCAN_OFFSETS)
        scatterProg = buildComputeProgram(Shaders.SCATTER)
        prepareProg = buildComputeProgram(Shaders.PREPARE)
        drawProg = buildDrawProgram(Shaders.DRAW_VERT, Shaders.DRAW_FRAG)

        GLES31.glGenBuffers(7, buffers, 0)
        allocBuffer(B.COUNTS, DEPTH_BUCKETS * 4)
        allocBuffer(B.CURSOR, DEPTH_BUCKETS * 4)
        allocBuffer(B.SCANNED, DEPTH_BUCKETS * 4)
        allocBuffer(B.BLOCKS, WORKGROUP * 4)

        GLES31.glDisable(GLES31.GL_DEPTH_TEST)
        GLES31.glDisable(GLES31.GL_CULL_FACE)
        GLES31.glEnable(GLES31.GL_BLEND)
        GLES31.glBlendFunc(GLES31.GL_ONE, GLES31.GL_ONE_MINUS_SRC_ALPHA)
        GLES31.glClearColor(0.04f, 0.04f, 0.06f, 1f)

        surfaceReady = true
        pendingLoad?.let { applyLoad(it.records, it.count, it.model, it.captureDist, it.resetCamera) }
        pendingLoad = null
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportW = width
        viewportH = height
        GLES31.glViewport(0, 0, width, height)
        Matrix.perspectiveM(
            projMatrix, 0, FOV_Y_DEG,
            width.toFloat() / height.toFloat().coerceAtLeast(1f),
            Z_NEAR, 60f
        )
        focal = 0.5f * height / tan(Math.toRadians((FOV_Y_DEG / 2f).toDouble()).toFloat())
        viewDirty = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT)
        if (splatCount == 0) {
            return
        }
        if (viewDirty) {
            runSort()
            viewDirty = false
        }
        GLES31.glUseProgram(drawProg)
        GLES31.glUniform2f(loc(drawProg, "uViewport"), viewportW.toFloat(), viewportH.toFloat())
        bindSlot(B.PREPARED, buffers[B.PREPARED])
        GLES31.glDrawArraysInstanced(GLES31.GL_TRIANGLE_STRIP, 0, 4, splatCount)
        framesDrawn++
    }

    fun loadSplat(records: ByteBuffer, count: Int, model: FloatArray, captureDist: Float, resetCamera: Boolean) {
        if (!surfaceReady) {
            pendingLoad = PendingLoad(records, count, model, captureDist, resetCamera)
            return
        }
        applyLoad(records, count, model, captureDist, resetCamera)
    }

    private fun applyLoad(records: ByteBuffer, count: Int, model: FloatArray, captureDist: Float, resetCamera: Boolean) {
        records.position(0)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[B.RECORDS])
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER, count * RECORD_BYTES,
            records, GLES31.GL_STATIC_DRAW
        )
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[B.ORDER])
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER, count * 4,
            null, GLES31.GL_DYNAMIC_DRAW
        )
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[B.PREPARED])
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER, count * PREPARED_BYTES,
            null, GLES31.GL_DYNAMIC_DRAW
        )
        System.arraycopy(model, 0, modelMatrix, 0, 16)
        splatCount = count
        lastSortCount = count
        if (resetCamera) {
            camera.reset(captureDist)
        }
        viewDirty = true
    }

    private fun runSort() {
        val t0 = SystemClock.uptimeMillis()
        zeroBuf.position(0)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[B.COUNTS])
        GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, DEPTH_BUCKETS * 4, zeroBuf)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[B.CURSOR])
        GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, DEPTH_BUCKETS * 4, zeroBuf)

        val n = splatCount
        val wg = (n + WORKGROUP - 1) / WORKGROUP
        val dist = camera.distance
        val near = (dist - 0.9f).coerceAtLeast(0.02f)
        val far = dist + 0.9f

        GLES31.glUseProgram(depthProg)
        setMat4(depthProg, "uModel", modelMatrix)
        setMat4(depthProg, "uView", camera.viewMatrix)
        GLES31.glUniform2f(loc(depthProg, "uDepthRange"), near, far)
        GLES31.glUniform1ui(loc(depthProg, "uCount"), n)
        bindSlot(B.RECORDS, buffers[B.RECORDS])
        bindSlot(B.COUNTS, buffers[B.COUNTS])
        GLES31.glDispatchCompute(wg, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        GLES31.glUseProgram(scanBlocksProg)
        bindSlot(B.COUNTS, buffers[B.COUNTS])
        bindSlot(B.SCANNED, buffers[B.SCANNED])
        bindSlot(B.BLOCKS, buffers[B.BLOCKS])
        GLES31.glDispatchCompute(SCAN_BLOCKS, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        GLES31.glUseProgram(scanSumsProg)
        bindSlot(B.BLOCKS, buffers[B.BLOCKS])
        GLES31.glDispatchCompute(1, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        GLES31.glUseProgram(scanOffsetsProg)
        bindSlot(B.SCANNED, buffers[B.SCANNED])
        bindSlot(B.BLOCKS, buffers[B.BLOCKS])
        bindSlot(B.COUNTS, buffers[B.COUNTS])
        GLES31.glDispatchCompute(SCAN_BLOCKS, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        GLES31.glUseProgram(scatterProg)
        setMat4(scatterProg, "uModel", modelMatrix)
        setMat4(scatterProg, "uView", camera.viewMatrix)
        GLES31.glUniform2f(loc(scatterProg, "uDepthRange"), near, far)
        GLES31.glUniform1ui(loc(scatterProg, "uCount"), n)
        bindSlot(B.RECORDS, buffers[B.RECORDS])
        bindSlot(B.COUNTS, buffers[B.COUNTS])
        bindSlot(B.CURSOR, buffers[B.CURSOR])
        bindSlot(B.ORDER, buffers[B.ORDER])
        GLES31.glDispatchCompute(wg, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        GLES31.glUseProgram(prepareProg)
        setMat4(prepareProg, "uModel", modelMatrix)
        setMat4(prepareProg, "uView", camera.viewMatrix)
        setMat4(prepareProg, "uProj", projMatrix)
        GLES31.glUniform2f(loc(prepareProg, "uViewport"), viewportW.toFloat(), viewportH.toFloat())
        GLES31.glUniform2f(loc(prepareProg, "uFocal"), focal, focal)
        GLES31.glUniform1ui(loc(prepareProg, "uCount"), n)
        bindSlot(B.RECORDS, buffers[B.RECORDS])
        bindSlot(B.ORDER, buffers[B.ORDER])
        bindSlot(B.PREPARED, buffers[B.PREPARED])
        GLES31.glDispatchCompute(wg, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_ALL_BARRIER_BITS)

        lastSortMs = SystemClock.uptimeMillis() - t0
    }

    private fun allocBuffer(idx: Int, sizeBytes: Int) {
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[idx])
        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, sizeBytes, null, GLES31.GL_DYNAMIC_DRAW)
    }

    private fun bindSlot(slot: Int, buf: Int) {
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, slot, buf)
    }

    private fun loc(prog: Int, name: String): Int = GLES31.glGetUniformLocation(prog, name)

    private fun setMat4(prog: Int, name: String, m: FloatArray) {
        GLES31.glUniformMatrix4fv(loc(prog, name), 1, false, m, 0)
    }

    private fun compile(type: Int, src: String): Int {
        val sh = GLES31.glCreateShader(type)
        GLES31.glShaderSource(sh, src)
        GLES31.glCompileShader(sh)
        val status = IntArray(1)
        GLES31.glGetShaderiv(sh, GLES31.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES31.glGetShaderInfoLog(sh)
            GLES31.glDeleteShader(sh)
            throw RuntimeException("Shader compile failed: $log")
        }
        return sh
    }

    private fun link(vararg shaders: Int): Int {
        val prog = GLES31.glCreateProgram()
        for (sh in shaders) {
            GLES31.glAttachShader(prog, sh)
        }
        GLES31.glLinkProgram(prog)
        val status = IntArray(1)
        GLES31.glGetProgramiv(prog, GLES31.GL_LINK_STATUS, status, 0)
        for (sh in shaders) {
            GLES31.glDetachShader(prog, sh)
            GLES31.glDeleteShader(sh)
        }
        if (status[0] == 0) {
            val log = GLES31.glGetProgramInfoLog(prog)
            GLES31.glDeleteProgram(prog)
            throw RuntimeException("Program link failed: $log")
        }
        return prog
    }

    private fun buildComputeProgram(src: String): Int {
        val sh = compile(GLES31.GL_COMPUTE_SHADER, src)
        return link(sh)
    }

    private fun buildDrawProgram(vertSrc: String, fragSrc: String): Int {
        val v = compile(GLES31.GL_VERTEX_SHADER, vertSrc)
        val f = compile(GLES31.GL_FRAGMENT_SHADER, fragSrc)
        return link(v, f)
    }
}
