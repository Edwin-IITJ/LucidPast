package com.lucidpast.ar.ui

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.lucidpast.ar.R
import java.io.File

class DockEntry(
    val id: String,
    val isVideo: Boolean,
    val thumbPath: String?,
)

class DockAdapter(
    private val root: File,
    private val entries: List<DockEntry>,
    private val onClick: (DockEntry) -> Unit,
) : RecyclerView.Adapter<DockAdapter.VH>() {

    private var selectedId: String? = null

    fun select(id: String) {
        selectedId = id
        notifyDataSetChanged()
    }

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val thumb: ImageView = v.findViewById(R.id.dock_thumb)
        val badge: TextView = v.findViewById(R.id.dock_badge)
        val card: View = v.findViewById(R.id.dock_card)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_dock, parent, false))

    override fun getItemCount(): Int = entries.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val e = entries[position]
        if (e.thumbPath != null) {
            val bmp = BitmapFactory.decodeFile(File(root, e.thumbPath).absolutePath)
            if (bmp != null) {
                holder.thumb.setImageBitmap(bmp)
            }
        }
        holder.badge.visibility = if (e.isVideo) View.VISIBLE else View.GONE
        holder.card.setBackgroundResource(
            if (e.id == selectedId) R.drawable.dock_card_bg_selected
            else R.drawable.dock_card_bg
        )
        holder.itemView.setOnClickListener { onClick(e) }
    }
}
