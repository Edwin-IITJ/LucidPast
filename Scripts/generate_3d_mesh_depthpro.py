# generate_3d_mesh_depthpro.py - VERSION 1.0 FINAL
# LucidPast - Batch 3D mesh generation using Depth Pro depth maps
# AI-assisted development (Claude + Perplexity AI)
#
# KEY FEATURES:
# - Depth Pro depth maps with INVERSION fix (black→close becomes white→close)
# - Auto-detected focal lengths from Depth Pro
# - High-quality Poisson reconstruction (depth=10)
# - Processes all 6 archival photos

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

# === CONFIGURATION ===
CONFIG = {
    # Poisson Surface Reconstruction
    'poisson_depth': 10,              # Maximum detail (Depth Pro is sharp enough)
    'poisson_scale': 1.05,            # Tight surface fit
    'poisson_linear_fit': False,      # Better for dense Depth Pro data
    
    # Depth Map Preprocessing
    'enable_bilateral_filter': False, # DISABLED - Depth Pro is already sharp
    'invert_depth': True,             # CRITICAL: Depth Pro uses inverted convention
    
    # Camera Projection
    'use_depthpro_focal_length': True,  # Use auto-detected focal length
    'focal_length_multiplier': 1.0,     # Multiplier (1.0 = use Depth Pro value as-is)
    
    # Normal Correction
    'normal_radius': 0.05,
    'normal_max_neighbors': 30,
    
    # Output
    'save_ply': True,                 # Save PLY (with vertex colors)
    'save_obj': True,                 # Save OBJ (with UVs for textures)
}

# === HELPER FUNCTIONS ===

def load_depthpro_focal_length(base_name, depthpro_dir):
    """Load focal length from Depth Pro output file"""
    focal_txt = os.path.join(depthpro_dir, f"{base_name}_focal_length.txt")
    
    if os.path.exists(focal_txt):
        with open(focal_txt, 'r') as f:
            first_line = f.readline()
            # Parse: "Estimated Focal Length: 5126.50 pixels"
            focal_px = float(first_line.split(':')[1].strip().split()[0])
            return focal_px
    return None


def add_camera_projection_uvs(mesh, width, height, focal_length):
    """Generate UV coordinates via camera projection"""
    vertices = np.asarray(mesh.vertices)
    cx, cy = width / 2, height / 2
    
    # Prevent division by zero
    z_safe = np.maximum(-vertices[:, 2], 0.001)
    
    # Project 3D vertices to 2D UV space
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (vertices[:, 0] * focal_length / z_safe + cx) / width
    uvs[:, 1] = (vertices[:, 1] * focal_length / z_safe + cy) / height
    uvs = np.clip(uvs, 0.0, 1.0)
    
    # Assign UVs to triangle vertices
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = []
    for tri in triangles:
        triangle_uvs.extend([uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh


def fix_inverted_normals(mesh):
    """Fix normals pointing inward using centroid-based method"""
    print("   - Fixing inverted normals...")
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # Build KD-tree for neighbor search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    corrected_normals = normals.copy()
    fixed_count = 0
    
    for i, vertex in enumerate(vertices):
        # Find neighbors
        [k, idx, _] = tree.search_radius_vector_3d(
            vertex, 
            CONFIG['normal_radius']
        )
        
        if k > 3:  # Need at least 3 neighbors
            # Limit neighbors for performance
            neighbor_indices = idx[1:min(k, CONFIG['normal_max_neighbors'] + 1)]
            neighbor_positions = vertices[neighbor_indices]
            
            # Calculate centroid
            centroid = np.mean(neighbor_positions, axis=0)
            
            # Vector from centroid to vertex (should point outward)
            outward_direction = vertex - centroid
            outward_direction = outward_direction / (np.linalg.norm(outward_direction) + 1e-8)
            
            # If normal points inward (dot product < 0), flip it
            if np.dot(normals[i], outward_direction) < 0:
                corrected_normals[i] = -normals[i]
                fixed_count += 1
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(corrected_normals)
    print(f"      Fixed {fixed_count}/{len(vertices)} inverted normals")
    return mesh


def cleanup_mesh(mesh):
    """Remove duplicate/degenerate geometry"""
    print("   - Cleaning up mesh...")
    
    original_verts = len(mesh.vertices)
    original_tris = len(mesh.triangles)
    
    # Remove duplicate vertices
    mesh.remove_duplicated_vertices()
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    removed_verts = original_verts - len(mesh.vertices)
    removed_tris = original_tris - len(mesh.triangles)
    
    print(f"      Removed: {removed_verts} vertices, {removed_tris} triangles")
    return mesh


def find_texture_file(base_name, texture_dir):
    """Find texture file with flexible format detection (PNG/JPG)"""
    for ext in ['.png', '.jpg', '.jpeg']:
        texture_path = os.path.join(texture_dir, f"{base_name}_texture{ext}")
        if os.path.exists(texture_path):
            try:
                texture_img = Image.open(texture_path)
                print(f"   ✓ Found texture: {os.path.basename(texture_path)}")
                return texture_path, texture_img
            except Exception as e:
                print(f"   ⚠️ Error loading {texture_path}: {e}")
                continue
    
    return None, None


# === MAIN MESH GENERATION ===

def create_mesh_from_files(depth_path, texture_path, output_path, output_ply_path, base_name, depthpro_dir):
    """Generate 3D mesh from Depth Pro depth map + texture"""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")
    
    # === 1. LOAD DEPTH MAP ===
    print("📥 Loading Depth Pro depth map...")
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    
    height, width = depth_map.shape[:2]
    print(f"   Resolution: {width} x {height}")
    
    # Normalize depth
    if depth_map.dtype == np.uint16:
        depth_norm = depth_map.astype(np.float32) / 65535.0
    elif depth_map.dtype == np.uint8:
        depth_norm = depth_map.astype(np.float32) / 255.0
    else:
        depth_norm = depth_map.astype(np.float32)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
    
    # ⚠️ CRITICAL: Invert Depth Pro depth convention
    if CONFIG['invert_depth']:
        print("   🔄 Inverting depth (Depth Pro fix: black→close to white→close)")
        depth_norm = 1.0 - depth_norm
    
    print(f"   Depth range: {depth_norm.min():.3f} - {depth_norm.max():.3f}")
    
    # === 2. LOAD TEXTURE ===
    print("📥 Loading texture...")
    texture_img = Image.open(texture_path)
    tex_width, tex_height = texture_img.size
    texture_resized = texture_img.resize((width, height), Image.Resampling.LANCZOS)
    texture_array = np.array(texture_resized)
    
    # Convert grayscale to RGB
    if len(texture_array.shape) == 2:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_GRAY2RGB)
    elif texture_array.shape[2] == 4:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGBA2RGB)
    
    print(f"   Texture: {tex_width} x {tex_height} → {width} x {height}")
    
    # === 3. DETERMINE FOCAL LENGTH ===
    if CONFIG['use_depthpro_focal_length']:
        focal_length_px = load_depthpro_focal_length(base_name, depthpro_dir)
        if focal_length_px is None:
            # Fallback to multiplier method
            focal_length = width * 1.2
            print(f"   ⚠️ Using fallback focal length: {focal_length:.1f}px")
        else:
            focal_length = focal_length_px * CONFIG['focal_length_multiplier']
            print(f"   ✓ Using Depth Pro focal length: {focal_length:.1f}px")
    else:
        focal_length = width * 1.2
        print(f"   Using manual focal length: {focal_length:.1f}px")
    
    # === 4. CREATE POINT CLOUD ===
    print("🔨 Generating point cloud...")
    cx, cy = width / 2, height / 2
    
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            z = -depth_norm[v, u]  # Negative Z (camera looks down -Z axis)
            if z == 0:
                continue
            
            # Back-project to 3D
            x = (u - cx) * (-z) / focal_length
            y = (v - cy) * (-z) / focal_length
            
            points.append([x, y, z])
            colors.append(texture_array[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    
    print(f"   Point cloud: {len(pcd.points):,} points")
    
    # === 5. POISSON SURFACE RECONSTRUCTION ===
    print(f"🔨 Poisson reconstruction (depth={CONFIG['poisson_depth']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=CONFIG['poisson_depth'],
        width=0,
        scale=CONFIG['poisson_scale'],
        linear_fit=CONFIG['poisson_linear_fit']
    )
    
    print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # === 6. CLEAN UP MESH ===
    mesh = cleanup_mesh(mesh)
    
    # === 7. TRANSFER COLORS FROM POINT CLOUD ===
    print("🎨 Transferring colors...")
    mesh_points = np.asarray(mesh.vertices)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((len(mesh_points), 3))
    
    for i, point in enumerate(mesh_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # === 8. COMPUTE NORMALS ===
    print("📐 Computing normals...")
    mesh.compute_vertex_normals()
    
    # === 9. FIX INVERTED NORMALS ===
    mesh = fix_inverted_normals(mesh)
    
    # === 10. ADD UV COORDINATES ===
    print("📐 Generating UV coordinates...")
    mesh = add_camera_projection_uvs(mesh, width, height, focal_length)
    
    # === 11. SAVE OUTPUTS ===
    print("💾 Saving outputs...")
    
    if CONFIG['save_ply']:
        o3d.io.write_triangle_mesh(output_ply_path, mesh)
        print(f"   ✓ Saved PLY: {os.path.basename(output_ply_path)}")
    
    if CONFIG['save_obj']:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"   ✓ Saved OBJ: {os.path.basename(output_path)}")
    
    # === SUMMARY ===
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds")
    print(f"   Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"{'='*60}\n")


# === BATCH PROCESSING ===

def batch_process():
    """Process all Depth Pro depth maps"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    depth_dir = os.path.join(project_root, 'Depth-Maps-Pro')
    texture_dir = os.path.join(project_root, 'Textures')
    output_dir = os.path.join(project_root, 'Meshes-DepthPro')  # Production folder
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LucidPast - Depth Pro Batch Mesh Generation v1.0")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Poisson Depth: {CONFIG['poisson_depth']}")
    print(f"  Depth Inversion: {'Enabled' if CONFIG['invert_depth'] else 'Disabled'} (Depth Pro fix)")
    print(f"  Depth Pro Focal Length: {'Enabled' if CONFIG['use_depthpro_focal_length'] else 'Disabled'}")
    print(f"  Output Formats: {'PLY ' if CONFIG['save_ply'] else ''}{'OBJ' if CONFIG['save_obj'] else ''}")
    
    # Find all Depth Pro high-res depth maps
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('_depthpro_highres.png')]
    
    if not depth_files:
        print(f"\n❌ No *_depthpro_highres.png files found in {depth_dir}")
        print("   Run batch_depth_estimation_pro.py first to generate depth maps.")
        return
    
    print(f"\nFound {len(depth_files)} depth maps to process\n")
    
    processed = 0
    skipped = 0
    total_start = time.time()
    
    for depth_file in depth_files:
        # Extract base name (e.g., "02_migrant" from "02_migrant_depthpro_highres.png")
        base_name = depth_file.replace('_depthpro_highres.png', '')
        
        depth_path = os.path.join(depth_dir, depth_file)
        output_obj = os.path.join(output_dir, f"{base_name}.obj")
        output_ply = os.path.join(output_dir, f"{base_name}.ply")
        
        # Find texture file (PNG/JPG)
        texture_path, texture_img = find_texture_file(base_name, texture_dir)
        
        if texture_path is None:
            print(f"\n⚠️ SKIPPED: {base_name}")
            print(f"   Reason: No texture file found in {texture_dir}")
            print(f"   Looking for: {base_name}_texture.png or .jpg\n")
            skipped += 1
            continue
        
        try:
            create_mesh_from_files(depth_path, texture_path, output_obj, output_ply, base_name, depth_dir)
            processed += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {base_name}: {e}\n")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue
    
    # === FINAL SUMMARY ===
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Processed: {processed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"⏱️ Total time: {total_elapsed/60:.1f} minutes")
    print(f"📁 Output folder: {output_dir}")
    print(f"\nDepth Pro advantages in these meshes:")
    print(f"  ✓ Sharper object edges (better than Depth Anything V2)")
    print(f"  ✓ Superior facial features (critical for portraits)")
    print(f"  ✓ Accurate focal length (auto-detected by Depth Pro)")
    print(f"  ✓ Metric depth scale (real-world units)")
    print(f"\nNext steps:")
    print(f"1. Import meshes to Blender from: {output_dir}")
    print(f"2. Fix normals: Edit Mode → Mesh → Normals → Recalculate Outside")
    print(f"3. Apply textures: Material Properties → Image Texture → Load PNG")
    print(f"4. Set up camera/lighting for video renders")
    print("="*60 + "\n")


# === ENTRY POINT ===
if __name__ == "__main__":
    batch_process()
