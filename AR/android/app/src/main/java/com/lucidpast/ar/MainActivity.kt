package com.lucidpast.ar

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.lucidpast.ar.render.SplatRenderer
import com.lucidpast.ar.ui.DockAdapter
import com.lucidpast.ar.ui.DockEntry
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var glView: GLSurfaceView
    private lateinit var renderer: SplatRenderer
    private lateinit var debugView: TextView
    private var dockAdapter: DockAdapter? = null
    private var assetRoot: File? = null
    private var manifest: SplatManifest? = null
    private var playToken = 0

    private val io = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var lastLoadMs = 0L
    private var lastFrames = 0
    private var lastFpsSample = 0L
    private var currentFps = 0f

    private val debugRunnable = object : Runnable {
        override fun run() {
            updateDebug()
            mainHandler.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        glView = findViewById(R.id.gl_surface)
        debugView = findViewById(R.id.debug_overlay)
        renderer = SplatRenderer()
        glView.setEGLContextClientVersion(3)
        glView.setRenderer(renderer)
        setupTouch()
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
        mainHandler.postDelayed(debugRunnable, 500)
        if (assetRoot == null) {
            refreshAssets()
        }
    }

    override fun onPause() {
        super.onPause()
        playToken++
        glView.onPause()
        mainHandler.removeCallbacks(debugRunnable)
    }

    private fun refreshAssets() {
        if (!Environment.isExternalStorageManager()) {
            debugView.text = getString(R.string.storage_permission)
            try {
                startActivity(
                    Intent(
                        Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                )
            } catch (e: Exception) {
                Toast.makeText(this, R.string.storage_permission, Toast.LENGTH_LONG).show()
            }
            return
        }
        val root = File(Environment.getExternalStorageDirectory(), "LucidPastAR")
        if (!File(root, "manifest.json").exists()) {
            debugView.text = getString(R.string.no_assets)
            return
        }
        assetRoot = root
        io.execute {
            var m: SplatManifest? = null
            var err: String? = null
            try {
                m = SplatManifest.load(root)
            } catch (e: Exception) {
                err = e.message
            }
            val loaded = m
            val error = err
            mainHandler.post {
                if (loaded == null) {
                    debugView.text = "manifest.json parse failed: $error"
                } else {
                    manifest = loaded
                    buildDock(loaded)
                }
            }
        }
    }

    private fun buildDock(m: SplatManifest) {
        val root = assetRoot ?: return
        val entries = ArrayList<DockEntry>()
        m.statics.forEach { entries.add(DockEntry(it.id, false, it.thumb)) }
        m.videos.forEach { entries.add(DockEntry(it.id, true, it.thumb)) }
        val view = findViewById<RecyclerView>(R.id.dock)
        view.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        val adapter = DockAdapter(root, entries) { entry -> onEntrySelected(entry) }
        view.adapter = adapter
        dockAdapter = adapter
        m.statics.firstOrNull()?.let {
            onEntrySelected(DockEntry(it.id, false, it.thumb))
        }
    }

    private fun onEntrySelected(entry: DockEntry) {
        val root = assetRoot ?: return
        val m = manifest ?: return
        val static = m.statics.find { it.id == entry.id }
        val video = m.videos.find { it.id == entry.id }
        val token = ++playToken

        if (video != null) {
            Toast.makeText(
                this,
                getString(R.string.video_frame_zero, video.id),
                Toast.LENGTH_SHORT
            ).show()
            val model = SplatIO.modelMatrix(video.center, video.extents, 0.45f)
            dockAdapter?.select(entry.id)
            io.execute {
                loadVideoFrame(root, video, 0, model, SplatRenderer.VIDEO_FIT_DIST, token)
            }
            return
        }

        val st = static ?: return
        val file = File(root, st.file)
        val center = st.center
        val extents = st.extents
        val maxExtent = maxOf(extents[0], extents[1], extents[2]).coerceAtLeast(1e-6f)
        val captureDist = ((0.45f / maxExtent) * abs(center[2])).coerceIn(0.25f, 4.0f)
        dockAdapter?.select(entry.id)
        io.execute {
            val t0 = System.currentTimeMillis()
            val loaded = try {
                SplatIO.readRecords(file)
            } catch (e: Exception) {
                null
            } ?: return@execute
            val loadMs = System.currentTimeMillis() - t0
            val model = SplatIO.modelMatrix(center, extents, 0.45f)
            glView.queueEvent {
                renderer.loadSplat(loaded.first, loaded.second, model, captureDist, true)
            }
            mainHandler.post {
                lastLoadMs = loadMs
                Toast.makeText(
                    this,
                    getString(R.string.loaded, entry.id, loaded.second, loadMs),
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun loadVideoFrame(
        root: File,
        video: VideoAsset,
        index: Int,
        model: FloatArray,
        captureDist: Float,
        token: Int
    ) {
        if (token != playToken) { return }
        val file = SplatIO.frameFile(root, video.framePattern, index)
        val t0 = System.currentTimeMillis()
        val loaded = try {
            SplatIO.readRecords(file)
        } catch (e: Exception) {
            null
        } ?: return
        val loadMs = System.currentTimeMillis() - t0
        glView.queueEvent {
            renderer.loadSplat(loaded.first, loaded.second, model, captureDist, index == 0)
        }
        mainHandler.post { lastLoadMs = loadMs }
        val interval = (1000.0 / video.fps).toLong().coerceAtLeast(16L)
        val next = (index + 1) % video.frameCount
        mainHandler.postDelayed({
            if (token == playToken) {
                io.execute { loadVideoFrame(root, video, next, model, captureDist, token) }
            }
        }, interval)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupTouch() {
        val scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    renderer.camera.zoom(1f / detector.scaleFactor)
                    renderer.viewDirty = true
                    return true
                }
            }
        )
        val gestureListener = object : GestureDetector.SimpleOnGestureListener() {
            override fun onScroll(
                e1: MotionEvent?,
                e2: MotionEvent,
                dx: Float,
                dy: Float
            ): Boolean {
                renderer.camera.drag(dx, dy)
                renderer.viewDirty = true
                return true
            }
        }
        val gestureDetector = GestureDetector(this, gestureListener)
        glView.setOnTouchListener { _, event ->
            scaleDetector.onTouchEvent(event)
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun updateDebug() {
        val now = System.currentTimeMillis()
        val dt = (now - lastFpsSample).coerceAtLeast(1)
        if (lastFpsSample > 0) {
            currentFps = (renderer.framesDrawn - lastFrames) * 1000f / dt
        }
        lastFrames = renderer.framesDrawn
        lastFpsSample = now
        debugView.text = String.format(
            Locale.US,
            "fps %.0f  |  splats %d  |  sort %d ms  |  load %d ms",
            currentFps, renderer.lastSortCount, renderer.lastSortMs, lastLoadMs
        )
    }
}
