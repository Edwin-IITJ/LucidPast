# generate_3d_mesh_textured_gltf.py - VERSION 6.0 FINAL PRODUCTION
# Uses converted PNG textures for mesh generation + texture mapping
# All fixes: Normal correction, cleanup, UVs

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

def add_camera_projection_uvs(mesh, width, height, focal_length):
    """Generate UVs via camera projection"""
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
    """Fix normals pointing inward"""
    print("→ Correcting inverted normals...")
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    if len(normals) == 0:
        print("  ⚠️  No normals")
        return mesh
    
    centroid = vertices.mean(axis=0)
    flipped = 0
    
    for i, (vertex, normal) in enumerate(zip(vertices, normals)):
        to_vertex = vertex - centroid
        if np.dot(normal, to_vertex) < 0:
            normals[i] = -normal
            flipped += 1
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    pct = flipped/len(normals)*100 if len(normals) > 0 else 0
    print(f"  ✓ Fixed {flipped:,} normals ({pct:.1f}%)")
    
    return mesh

def advanced_mesh_cleanup(mesh):
    """Remove artifacts and invalid geometry"""
    print("→ Mesh cleanup...")
    
    initial_verts = len(mesh.vertices)
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    # Remove disconnected components
    try:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 1:
            largest_idx = cluster_n_triangles.argmax()
            mask = triangle_clusters != largest_idx
            mesh.remove_triangles_by_mask(mask)
            mesh.remove_unreferenced_vertices()
            print(f"  • Removed {len(cluster_n_triangles)-1} components")
    except:
        pass
    
    final_verts = len(mesh.vertices)
    print(f"  ✓ {final_verts:,} verts ({initial_verts-final_verts:,} removed)")
    
    return mesh

def create_textured_mesh(depth_path, texture_path, output_base_path, focal_multiplier=1.2):
    """
    Generate mesh from depth + PNG texture
    Uses PNG for both RGBD creation AND texture reference
    """
    print(f"\n{'='*70}")
    print(f"MESH: {os.path.basename(output_base_path)}")
    print('='*70)
    
    # Load PNG texture
    print(f"→ Texture: {os.path.basename(texture_path)}")
    rgb_image = Image.open(texture_path).convert("RGB")
    rgb_np = np.array(rgb_image)
    height, width = rgb_np.shape[:2]
    print(f"  ✓ {width} x {height} px")
    
    # Load depth
    print("→ Depth map...")
    depth_path_hr = depth_path.replace('_depth.png', '_depth_highres.png')
    if os.path.exists(depth_path_hr):
        depth = np.array(Image.open(depth_path_hr).convert("L"))
        print("  ✓ High-res")
    else:
        depth = np.array(Image.open(depth_path).convert("L"))
        print("  ✓ Standard")
    
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    # Camera
    focal_length = width * focal_multiplier
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, cx, cy)
    print(f"→ Camera: focal={focal_length:.1f}")
    
    # RGBD
    rgb_o3d = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=255.0, depth_trunc=255.0, convert_rgb_to_intensity=False
    )
    
    # Point cloud
    print("→ Point cloud...")
    start = time.time()
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    print(f"  ✓ {len(pcd.points):,} points ({time.time()-start:.1f}s)")
    
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # Normals
    print("→ Normals...")
    start = time.time()
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    print(f"  ✓ ({time.time()-start:.1f}s)")
    
    # Poisson
    print("→ Poisson (depth=10, 2-5 min)...")
    start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, width=0, scale=1.1, linear_fit=True
    )
    print(f"  ✓ {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris ({time.time()-start:.1f}s)")
    
    # Remove outliers
    vertices_to_remove = densities < np.quantile(densities, 0.005)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Cleanup
    mesh = advanced_mesh_cleanup(mesh)
    
    # Smoothing
    print("→ Smoothing...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
    
    # Vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_np.reshape(-1, 3) / 255.0)
    
    # Normals
    mesh.compute_vertex_normals()
    mesh = fix_inverted_normals(mesh)
    
    # UVs
    print("→ UVs...")
    mesh = add_camera_projection_uvs(mesh, width, height, focal_length)
    
    # Export OBJ with MTL
    print("→ Export OBJ...")
    obj_path = output_base_path + ".obj"
    o3d.io.write_triangle_mesh(
        obj_path, mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True
    )
    print(f"  ✓ {os.path.basename(obj_path)}")
    
    # PLY
    ply_path = output_base_path + ".ply"
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)
    print(f"  ✓ {os.path.basename(ply_path)}")
    
    print(f"\n✅ COMPLETE!")
    print(f"  📊 {len(mesh.vertices):,} verts | {len(mesh.triangles):,} tris")
    print('='*70 + '\n')

if __name__ == "__main__":
    depth_dir = "../Depth-Maps"
    texture_dir = "../Textures"
    output_dir = "../Meshes"
    
    os.makedirs(output_dir, exist_ok=True)
    
    photos = [
        "01_diner_1940",
        "01_diner_1940_2",
        "02_migrant",
        "03_radio_studio_1942"
    ]
    
    print("\n" + "="*70)
    print("🚀 LUCIDPAST v6.0 - PNG-BASED MESH GENERATION")
    print("="*70)
    print(f"📁 Depth: {depth_dir}")
    print(f"📁 Textures: {texture_dir} (PNG)")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Files: {len(photos)}")
    print("="*70 + "\n")
    
    processed, errors, skipped = 0, 0, 0
    total_start = time.time()
    
    for i, base_name in enumerate(photos, 1):
        print(f"{'#'*70}")
        print(f"# FILE {i}/{len(photos)}: {base_name}")
        print(f"{'#'*70}\n")
        
        depth_path = os.path.join(depth_dir, f"{base_name}_depth.png")
        texture_path = os.path.join(texture_dir, f"{base_name}_texture.png")
        output_base = os.path.join(output_dir, base_name)
        
        if not os.path.exists(depth_path):
            print(f"⚠️  SKIP: No depth\n")
            skipped += 1
            continue
        
        if not os.path.exists(texture_path):
            print(f"⚠️  SKIP: No texture PNG\n")
            skipped += 1
            continue
        
        try:
            file_start = time.time()
            create_textured_mesh(depth_path, texture_path, output_base)
            print(f"⏱️  {(time.time()-file_start)/60:.1f} min\n")
            processed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            errors += 1
    
    print("\n" + "="*70)
    print("🏁 DONE!")
    print("="*70)
    print(f"✅ {processed}/{len(photos)}")
    print(f"⏱️  {(time.time()-total_start)/60:.1f} min")
    if skipped: print(f"⚠️  Skipped: {skipped}")
    if errors: print(f"❌ Errors: {errors}")
    print(f"\n📁 {os.path.abspath(output_dir)}")
    print("="*70 + "\n")
