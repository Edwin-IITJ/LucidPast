package com.lucidpast.ar

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class StaticAsset(
    val id: String,
    val file: String,
    val splatCount: Int,
    val center: FloatArray,
    val extents: FloatArray,
    val thumb: String?,
)

data class VideoAsset(
    val id: String,
    val fps: Double,
    val frameCount: Int,
    val framePattern: String,
    val splatCountPerFrame: Int,
    val center: FloatArray,
    val extents: FloatArray,
    val thumb: String?,
)

class SplatManifest private constructor(
    val statics: List<StaticAsset>,
    val videos: List<VideoAsset>,
) {
    companion object {
        fun load(root: File): SplatManifest {
            val json = JSONObject(File(root, "manifest.json").readText())

            fun percentiles(j: JSONObject): Pair<FloatArray, FloatArray>? {
                val p01 = j.optJSONArray("bbox_p01") ?: return null
                val p99 = j.optJSONArray("bbox_p99") ?: return null
                if (p01.length() != 3 || p99.length() != 3) return null
                val center = FloatArray(3) { (p01.getDouble(it).toFloat() + p99.getDouble(it).toFloat()) / 2f }
                val extents = FloatArray(3) { p99.getDouble(it).toFloat() - p01.getDouble(it).toFloat() }
                return Pair(center, extents)
            }

            fun centerOf(j: JSONObject): FloatArray {
                percentiles(j)?.let { return it.first }
                val c = j.getJSONArray("bbox_center")
                return FloatArray(3) { c.getDouble(it).toFloat() }
            }

            fun extentsOf(j: JSONObject): FloatArray {
                percentiles(j)?.let { return it.second }
                val mn = j.getJSONArray("bbox_min")
                val mx = j.getJSONArray("bbox_max")
                return FloatArray(3) { mx.getDouble(it).toFloat() - mn.getDouble(it).toFloat() }
            }

            fun thumbOf(j: JSONObject): String? =
                if (j.has("thumb") && !j.isNull("thumb")) j.getString("thumb") else null

            val statics = ArrayList<StaticAsset>()
            val js: JSONArray = json.optJSONArray("statics") ?: JSONArray()
            for (i in 0 until js.length()) {
                val j = js.getJSONObject(i)
                statics.add(
                    StaticAsset(
                        id = j.getString("id"),
                        file = j.getString("file"),
                        splatCount = j.getInt("splat_count"),
                        center = centerOf(j),
                        extents = extentsOf(j),
                        thumb = thumbOf(j),
                    )
                )
            }

            val videos = ArrayList<VideoAsset>()
            val jv: JSONArray = json.optJSONArray("videos") ?: JSONArray()
            for (i in 0 until jv.length()) {
                val j = jv.getJSONObject(i)
                videos.add(
                    VideoAsset(
                        id = j.getString("id"),
                        fps = j.optDouble("fps", 10.0),
                        frameCount = j.getInt("frame_count"),
                        framePattern = j.getString("frame_pattern"),
                        splatCountPerFrame = j.optInt("splat_count_per_frame", 0),
                        center = centerOf(j),
                        extents = extentsOf(j),
                        thumb = thumbOf(j),
                    )
                )
            }
            return SplatManifest(statics, videos)
        }
    }
}

object SplatIO {
    const val RECORD_BYTES = 32

    fun readRecords(file: File): Pair<ByteBuffer, Int> {
        val bytes = file.readBytes()
        val buf = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buf.put(bytes)
        buf.flip()
        return Pair(buf, bytes.size / RECORD_BYTES)
    }

    fun frameFile(root: File, pattern: String, index: Int): File =
        File(root, String.format(java.util.Locale.US, pattern, index))

    fun modelMatrix(center: FloatArray, extents: FloatArray, targetExtent: Float): FloatArray {
        val maxExtent = maxOf(extents[0], extents[1], extents[2]).coerceAtLeast(1e-6f)
        val s = targetExtent / maxExtent
        val m = FloatArray(16)
        m[0] = s
        m[5] = -s
        m[10] = -s
        m[12] = -s * center[0]
        m[13] = s * center[1]
        m[14] = s * center[2]
        m[15] = 1f
        return m
    }
}
