# generate_mesh_single_depthpro_fixed.py - FIXED DEPTH INVERSION
# Test on Migrant Mother with corrected Depth Pro depth convention

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

# === CONFIGURATION ===
CONFIG = {
    'poisson_depth': 10,
    'poisson_scale': 1.05,
    'poisson_linear_fit': False,
    'enable_bilateral_filter': False,
    'use_depthpro_focal_length': True,
    'focal_length_multiplier': 1.0,
    'normal_radius': 0.05,
    'normal_max_neighbors': 30,
}

IMAGE_NAME = "02_migrant"

# === HELPER FUNCTIONS ===
def load_depthpro_focal_length(base_name, depthpro_dir):
    focal_txt = os.path.join(depthpro_dir, f"{base_name}_focal_length.txt")
    if os.path.exists(focal_txt):
        with open(focal_txt, 'r') as f:
            first_line = f.readline()
            focal_px = float(first_line.split(':')[1].strip().split()[0])
            print(f"   ✓ Loaded Depth Pro focal length: {focal_px:.1f}px")
            return focal_px
    return None

def add_camera_projection_uvs(mesh, width, height, focal_length):
    vertices = np.asarray(mesh.vertices)
    cx, cy = width / 2, height / 2
    z_safe = np.maximum(-vertices[:, 2], 0.001)
    
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (vertices[:, 0] * focal_length / z_safe + cx) / width
    uvs[:, 1] = (vertices[:, 1] * focal_length / z_safe + cy) / height
    uvs = np.clip(uvs, 0.0, 1.0)
    
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = []
    for tri in triangles:
        triangle_uvs.extend([uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh

def fix_inverted_normals(mesh):
    print("   - Fixing inverted normals...")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    corrected_normals = normals.copy()
    fixed_count = 0
    
    for i, vertex in enumerate(vertices):
        [k, idx, _] = tree.search_radius_vector_3d(vertex, CONFIG['normal_radius'])
        if k > 3:
            neighbor_indices = idx[1:min(k, CONFIG['normal_max_neighbors'] + 1)]
            neighbor_positions = vertices[neighbor_indices]
            centroid = np.mean(neighbor_positions, axis=0)
            outward_direction = vertex - centroid
            outward_direction = outward_direction / (np.linalg.norm(outward_direction) + 1e-8)
            
            if np.dot(normals[i], outward_direction) < 0:
                corrected_normals[i] = -normals[i]
                fixed_count += 1
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(corrected_normals)
    print(f"      Fixed {fixed_count}/{len(vertices)} inverted normals")
    return mesh

# === MAIN PROCESSING ===
def generate_mesh():
    start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    depth_dir = os.path.join(project_root, 'Depth-Maps-Pro')
    texture_dir = os.path.join(project_root, 'Textures')
    output_dir = os.path.join(project_root, 'Meshes-DepthPro-Test')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Single Image Test: {IMAGE_NAME} (FIXED DEPTH INVERSION)")
    print("="*60)
    
    # File paths
    depth_path = os.path.join(depth_dir, f"{IMAGE_NAME}_depthpro_highres.png")
    texture_path_png = os.path.join(texture_dir, f"{IMAGE_NAME}_texture.png")
    texture_path_jpg = os.path.join(texture_dir, f"{IMAGE_NAME}_texture.jpg")
    
    if os.path.exists(texture_path_png):
        texture_path = texture_path_png
    elif os.path.exists(texture_path_jpg):
        texture_path = texture_path_jpg
    else:
        print(f"❌ Texture not found for {IMAGE_NAME}")
        return
    
    if not os.path.exists(depth_path):
        print(f"❌ Depth map not found: {depth_path}")
        return
    
    # === 1. LOAD DEPTH MAP ===
    print("📥 Loading Depth Pro depth map...")
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    height, width = depth_map.shape[:2]
    print(f"   Resolution: {width} x {height}")
    
    # Normalize depth
    if depth_map.dtype == np.uint16:
        depth_norm = depth_map.astype(np.float32) / 65535.0
    else:
        depth_norm = depth_map.astype(np.float32) / 255.0
    
    print(f"   Depth range (before inversion): {depth_norm.min():.3f} - {depth_norm.max():.3f}")
    
    # ⚠️ CRITICAL FIX: Depth Pro uses INVERTED depth convention
    # Depth Pro: Black = close, White = far
    # Our script expects: White = close, Black = far
    print("   🔄 Inverting Depth Pro depth map...")
    depth_norm = 1.0 - depth_norm  # Invert
    
    print(f"   Depth range (after inversion): {depth_norm.min():.3f} - {depth_norm.max():.3f}")
    print("   ✓ Now: White = close, Black = far (correct convention)")
    
    # === 2. LOAD TEXTURE ===
    print("📥 Loading texture...")
    texture_img = Image.open(texture_path)
    tex_width, tex_height = texture_img.size
    texture_resized = texture_img.resize((width, height), Image.Resampling.LANCZOS)
    texture_array = np.array(texture_resized)
    
    if len(texture_array.shape) == 2:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_GRAY2RGB)
    elif texture_array.shape[2] == 4:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGBA2RGB)
    
    print(f"   Texture: {tex_width} x {tex_height} → {width} x {height}")
    
    # === 3. LOAD FOCAL LENGTH ===
    if CONFIG['use_depthpro_focal_length']:
        focal_length_px = load_depthpro_focal_length(IMAGE_NAME, depth_dir)
        if focal_length_px:
            focal_length = focal_length_px * CONFIG['focal_length_multiplier']
            print(f"   Using Depth Pro focal length: {focal_length:.1f}px")
        else:
            focal_length = width * 1.2
            print(f"   Using fallback focal length: {focal_length:.1f}px")
    else:
        focal_length = width * 1.2
    
    # === 4. CREATE POINT CLOUD ===
    print("🔨 Generating point cloud...")
    cx, cy = width / 2, height / 2
    
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            z = -depth_norm[v, u]
            if z == 0:
                continue
            
            x = (u - cx) * (-z) / focal_length
            y = (v - cy) * (-z) / focal_length
            
            points.append([x, y, z])
            colors.append(texture_array[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    
    print(f"   Point cloud: {len(pcd.points):,} points")
    
    # === 5. POISSON RECONSTRUCTION ===
    print(f"🔨 Poisson reconstruction (depth={CONFIG['poisson_depth']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=CONFIG['poisson_depth'],
        width=0,
        scale=CONFIG['poisson_scale'],
        linear_fit=CONFIG['poisson_linear_fit']
    )
    
    print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # === 6. CLEANUP ===
    print("   - Cleaning up mesh...")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    
    # === 7. TRANSFER COLORS ===
    print("🎨 Transferring colors...")
    mesh_points = np.asarray(mesh.vertices)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((len(mesh_points), 3))
    
    for i, point in enumerate(mesh_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # === 8. NORMALS ===
    print("📐 Computing normals...")
    mesh.compute_vertex_normals()
    mesh = fix_inverted_normals(mesh)
    
    # === 9. UV MAPPING ===
    print("📐 Generating UV coordinates...")
    mesh = add_camera_projection_uvs(mesh, width, height, focal_length)
    
    # === 10. SAVE ===
    print("💾 Saving outputs...")
    output_obj = os.path.join(output_dir, f"{IMAGE_NAME}_depthpro_fixed.obj")
    output_ply = os.path.join(output_dir, f"{IMAGE_NAME}_depthpro_fixed.ply")
    
    o3d.io.write_triangle_mesh(output_ply, mesh)
    print(f"   ✓ Saved PLY: {os.path.basename(output_ply)}")
    
    o3d.io.write_triangle_mesh(output_obj, mesh)
    print(f"   ✓ Saved OBJ: {os.path.basename(output_obj)}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds")
    print(f"   Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"\n📁 Output: {output_obj}")
    print("\n🎉 Depth inversion fixed - mesh should now be correct!")
    print("="*60)

if __name__ == "__main__":
    generate_mesh()
