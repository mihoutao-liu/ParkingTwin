# -*- coding: utf-8 -*-
"""
Osm2Tsdf.py — Generate TSDF voxels and a triangle mesh from OSM data

Features:
- Parse OSM and classify geometries by sType (888 outer walls, 1000 inner walls/pillars, 1002 parking slots)
- Estimate dominant map orientation and scale (meters per OSM unit)
- Rotate/scale geometries into metric coordinates
- Build 2.5D TSDF in the XY plane and extrude along Z
- Optionally export a mesh via marching cubes and write world_from_osm.json for alignment

Required deps: numpy, shapely (>=1.8); Optional: scikit-image (marching cubes, EDT), matplotlib (preview), open3d (viz)

Example:
    python Osm2Tsdf.py \
        --osm ICPARK.osm \
        --outdir output/ICPARKOSM_generated \
        --stype-key sType \
        --voxel 0.10 \
        --height 3.0 \
        --trunc 0.40
"""
from __future__ import annotations
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from xml.etree import ElementTree as ET

try:
    import shapely
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
    from shapely.ops import unary_union, linemerge, polygonize, triangulate
    from shapely.affinity import rotate as shp_rotate, scale as shp_scale, translate as shp_translate
    from shapely.geometry.polygon import orient as shp_orient
except Exception as e:
    raise RuntimeError("shapely required: pip install shapely") from e

# Optional dependencies
try:
    from skimage.measure import marching_cubes
    from skimage.morphology import distance_transform_edt as sk_edt
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Utility functions and data structures

def np_to_list(arr: np.ndarray):
    return [float(x) for x in np.asarray(arr).reshape(-1)]

@dataclass
class WorldTransform:
    applied_rotation_deg: float
    centroid_osm: Tuple[float, float]
    scale_m_per_osm: float
    prior_tsdf_origin_xyz: Tuple[float, float, float]

# OSM parsing

def load_osm(osm_path: str, stype_key: str = "sType"):
    """Parse OSM file and return node dictionary and ways list classified by sType.
    Returns: nodes: Dict[int,(x,y)], ways: List[dict(id, stype, node_ids, is_closed)]
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: Dict[int, Tuple[float, float]] = {}

    for n in root.findall('node'):
        nid = int(n.get('id'))
        x = float(n.get('lon'))
        y = float(n.get('lat'))
        nodes[nid] = (x, y)

    ways: List[dict] = []
    for w in root.findall('way'):
        wid = int(w.get('id'))
        nds = [int(nd.get('ref')) for nd in w.findall('nd')]
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        stype = int(tags.get(stype_key, -1)) if tags.get(stype_key) is not None else -1
        is_closed = (len(nds) >= 3 and nds[0] == nds[-1])
        ways.append(dict(id=wid, stype=stype, node_ids=nds, is_closed=is_closed))

    return nodes, ways

# Geometry construction and cleanup

def build_geometries(nodes: Dict[int, Tuple[float, float]], ways: List[dict]):
    """Classify geometries by sType. Returns dict:
        {
          888: [LineString, ...],  # outer walls
          1000: {"polys": [Polygon,...], "lines": [LineString,...]},  # inner walls/pillars
          1002: [Polygon,...]  # parking slots
        }
    """
    walls: List[LineString] = []
    inner_polys: List[Polygon] = []
    inner_lines: List[LineString] = []
    slots: List[Polygon] = []

    def nodes_to_xy(seq: List[int]):
        return [nodes[i] for i in seq if i in nodes]

    for w in ways:
        st = w['stype']
        pts = nodes_to_xy(w['node_ids'])
        if len(pts) < 2:
            continue
        if st == 888:
            walls.append(LineString(pts))
        elif st == 1000:
            if w['is_closed'] or (np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 1e-6):
                try:
                    poly = Polygon(pts)
                    if poly.is_valid and not poly.is_empty and poly.area > 0:
                        inner_polys.append(shp_orient(poly, sign=1.0))
                    else:
                        inner_lines.append(LineString(pts))
                except Exception:
                    inner_lines.append(LineString(pts))
            else:
                inner_lines.append(LineString(pts))
        elif st == 1002:
            try:
                poly = Polygon(pts)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    slots.append(shp_orient(poly, sign=1.0))
            except Exception:
                pass

    geoms = {
        888: walls,
        1000: {"polys": inner_polys, "lines": inner_lines},
        1002: slots
    }
    return geoms

# Dominant orientation and scale estimation

def line_directions(geom_list: List[LineString]) -> List[float]:
    """Sample direction angles from lines (degrees, folded to [-90,90) for dominant direction)."""
    dirs: List[float] = []
    for ln in geom_list:
        if ln is None or ln.is_empty:
            continue
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            dx, dy = (x1 - x0), (y1 - y0)
            if abs(dx) + abs(dy) < 1e-9:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            ang = ((ang + 180) % 180) - 180
            if ang >= 90:
                ang -= 180
            if ang < -90:
                ang += 180
            dirs.append(ang)
    return dirs


def polygon_edge_directions(polys: List[Polygon]) -> List[float]:
    dirs: List[float] = []
    for p in polys:
        if p is None or p.is_empty:
            continue
        coords = list(p.exterior.coords)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            dx, dy = (x1 - x0), (y1 - y0)
            if abs(dx) + abs(dy) < 1e-9:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            ang = ((ang + 180) % 180) - 180
            if ang >= 90:
                ang -= 180
            if ang < -90:
                ang += 180
            dirs.append(ang)
    return dirs


def dominant_angle_deg(geoms: dict) -> float:
    """Estimate dominant map orientation (0~90 degrees), returns rotation angle to apply to OSM (negative to align with X-axis)."""
    cand_dirs: List[float] = []
    cand_dirs += line_directions(geoms.get(888, []))
    cand_dirs += line_directions(geoms.get(1000, {}).get('lines', []))
    cand_dirs += polygon_edge_directions(geoms.get(1000, {}).get('polys', []))
    cand_dirs += polygon_edge_directions(geoms.get(1002, []))
    if len(cand_dirs) == 0:
        return 0.0

    dirs = np.abs(np.array(cand_dirs))
    hist, edges = np.histogram(dirs, bins=90, range=(0, 90))
    idx = int(np.argmax(hist))
    center = 0.5 * (edges[idx] + edges[idx + 1])
    return -float(center)


def estimate_scale_from_slots(slots: List[Polygon], target_width_m: float = 2.5) -> Optional[float]:
    """Estimate OSM unit to meter scale from parking slot minimum bounding rectangles (using short edge as width). Returns None if insufficient samples."""
    widths = []
    for p in slots:
        try:
            mr = p.minimum_rotated_rectangle
            coords = list(mr.exterior.coords)
            if len(coords) < 5:
                continue
            edges = [np.linalg.norm(np.array(coords[i]) - np.array(coords[(i + 1) % 4])) for i in range(4)]
            w = min(edges)
            if w > 1e-6:
                widths.append(w)
        except Exception:
            pass
    if len(widths) < 5:
        return None
    median_w = float(np.median(widths))
    return float(target_width_m / median_w) if median_w > 0 else None

# Affine transformation: rotation + scale + translation

def apply_rotation_and_scale(geoms: dict, centroid: Tuple[float, float], rot_deg: float, scale_m_per_osm: float):
    cx, cy = centroid
    def aff(g):
        if g is None or g.is_empty:
            return g
        g2 = shp_translate(g, xoff=-cx, yoff=-cy)
        g2 = shp_rotate(g2, rot_deg, origin=(0, 0), use_radians=False)
        g2 = shp_scale(g2, xfact=scale_m_per_osm, yfact=scale_m_per_osm, origin=(0, 0))
        g2 = shp_translate(g2, xoff=cx * scale_m_per_osm, yoff=cy * scale_m_per_osm)
        return g2

    geoms2 = {
        888: [aff(g) for g in geoms.get(888, [])],
        1000: {
            'polys': [aff(g) for g in geoms.get(1000, {}).get('polys', [])],
            'lines': [aff(g) for g in geoms.get(1000, {}).get('lines', [])],
        },
        1002: [aff(g) for g in geoms.get(1002, [])]
    }
    return geoms2

# TSDF generation (2.5D)

def rasterize_geometry(geoms_m: dict, voxel: float, margin_vox: int = 4):
    """Project geometries to 2D raster grid. Returns mask (obstacle=1), bounds, origin.
    - Outer/inner walls: drawn with buffer thickness (default 0.1m)
    - Pillars/closed polygons: directly filled
    - Parking slots: used for scale estimation only, not added to mask
    """
    wall_thickness = max(0.10, 1.5 * voxel)

    walls = geoms_m.get(888, [])
    inner_lines = geoms_m.get(1000, {}).get('lines', [])
    inner_polys = geoms_m.get(1000, {}).get('polys', [])
    bounds = None
    def update_bounds(b, geom):
        if geom is None or geom.is_empty:
            return b
        minx, miny, maxx, maxy = geom.bounds
        if b is None:
            return (minx, miny, maxx, maxy)
        return (min(b[0], minx), min(b[1], miny), max(b[2], maxx), max(b[3], maxy))

    for lst in [walls, inner_lines, inner_polys]:
        for g in lst:
            bounds = update_bounds(bounds, g)

    if bounds is None:
        raise RuntimeError("Empty geometry, cannot generate TSDF")

    minx, miny, maxx, maxy = bounds
    minx -= margin_vox * voxel
    miny -= margin_vox * voxel
    maxx += margin_vox * voxel
    maxy += margin_vox * voxel

    width = maxx - minx
    height = maxy - miny
    nx = int(math.ceil(width / voxel))
    ny = int(math.ceil(height / voxel))

    mask = np.zeros((ny, nx), dtype=np.uint8)

    def draw_polygon_to_mask(poly: Polygon, value: int = 1):
        if poly is None or poly.is_empty:
            return
        minx_p, miny_p, maxx_p, maxy_p = poly.bounds
        ix0 = max(0, int((minx_p - minx) / voxel))
        iy0 = max(0, int((miny_p - miny) / voxel))
        ix1 = min(nx - 1, int((maxx_p - minx) / voxel))
        iy1 = min(ny - 1, int((maxy_p - miny) / voxel))
        xs = minx + (np.arange(ix0, ix1 + 1) + 0.5) * voxel
        ys = miny + (np.arange(iy0, iy1 + 1) + 0.5) * voxel
        for iy, y in enumerate(ys, start=iy0):
            for ix, x in enumerate(xs, start=ix0):
                if poly.contains(Point(x, y)) or poly.touches(Point(x, y)):
                    mask[iy, ix] = value
    line_polys: List[Polygon] = []
    for ln in walls + inner_lines:
        try:
            bp = ln.buffer(wall_thickness / 2.0, cap_style=2, join_style=2)
            if bp and not bp.is_empty:
                if isinstance(bp, Polygon):
                    line_polys.append(bp)
                elif isinstance(bp, MultiPolygon):
                    line_polys.extend(list(bp.geoms))
        except Exception:
            pass

    all_polys: List[Polygon] = []
    all_polys.extend(inner_polys)
    all_polys.extend(line_polys)

    merged = None
    if len(all_polys) > 0:
        try:
            merged = unary_union(all_polys)
        except Exception:
            merged = None

    def handle_poly_geom(geom):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            draw_polygon_to_mask(geom, 1)
        elif isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                draw_polygon_to_mask(g, 1)

    handle_poly_geom(merged)

    origin = (minx, miny, 0.0)
    dims = (nx, ny)
    return mask, origin, dims


def compute_tsdf_from_mask(mask: np.ndarray, voxel: float, height: float, trunc: float):
    """Compute signed distance on XY plane, then replicate along Z to form 3D voxels.
    TSDF values are approximately (-trunc, +trunc), clipped to [-trunc, trunc].
    Returns: tsdf(Hz,Hy,Hx), voxel_size=(vx,vy,vz), dims=(nx,ny,nz)
    """
    ny, nx = mask.shape

    if SKIMAGE_AVAILABLE:
        dist_out = sk_edt(1 - mask)
        dist_in = sk_edt(mask)
    else:
        from collections import deque
        def bfs_dt(bin_img, value):
            H, W = bin_img.shape
            INF = 109
            dist = np.full((H, W), INF, dtype=np.int32)
            q = deque()
            ys, xs = np.where(bin_img == value)
            for y, x in zip(ys, xs):
                dist[y, x] = 0
                q.append((y, x))
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
            while q:
                y, x = q.popleft()
                d0 = dist[y, x]
                for dy, dx in dirs:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and dist[yy, xx] == INF:
                        dist[yy, xx] = d0 + 1
                        q.append((yy, xx))
            return dist.astype(np.float32)
        dist_out = bfs_dt(mask, 1)
        dist_in  = bfs_dt(1 - mask, 1)
    dist_out_m = dist_out * voxel
    dist_in_m = dist_in * voxel

    sdf_2d = dist_out_m.copy()
    sdf_2d[mask == 1] = -dist_in_m[mask == 1]
    sdf_2d = np.clip(sdf_2d, -trunc, trunc)

    nz = max(1, int(math.ceil(height / voxel)))
    tsdf = np.repeat(sdf_2d[np.newaxis, :, :], nz, axis=0)  # (z,y,x)

    voxel_size = (voxel, voxel, voxel)
    dims3 = (nx, ny, nz)
    return tsdf.astype(np.float32), voxel_size, dims3

# Meshing and export

def export_ply_mesh_from_tsdf(tsdf: np.ndarray, voxel: float, origin: Tuple[float, float, float], out_path: str):
    """Extract zero-level isosurface using marching cubes and export PLY. Skips if skimage unavailable."""
    if not SKIMAGE_AVAILABLE:
        print("[WARN] scikit-image not installed, skipping mesh export:", out_path)
        return False

    vol = tsdf
    verts, faces, normals, values = marching_cubes(vol, level=0.0, spacing=(voxel, voxel, voxel))
    verts[:, 0] += origin[2]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[0]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[2]:.6f} {v[1]:.6f} {v[0]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[2]} {tri[1]} {tri[0]}\n")
    print("[OK] 导出网格：", out_path)
    return True


def save_boundary_and_world(outdir: str, world: WorldTransform):
    bcfg = os.path.join(outdir, 'boundary_config.txt')
    with open(bcfg, 'w', encoding='utf-8') as f:
        f.write("# prior tsdf origin xyz (meters)\n")
        f.write("prior_tsdf_origin_xyz = %.6f, %.6f, %.6f\n" % world.prior_tsdf_origin_xyz)
    print("[OK] Written:", bcfg)
    wj = os.path.join(outdir, 'world_from_osm.json')
    with open(wj, 'w', encoding='utf-8') as f:
        json.dump({
            'applied_rotation_deg': float(world.applied_rotation_deg),
            'centroid_osm': [float(world.centroid_osm[0]), float(world.centroid_osm[1])],
            'scale_m_per_osm': float(world.scale_m_per_osm),
            'prior_tsdf_origin_xyz': [float(world.prior_tsdf_origin_xyz[0]), float(world.prior_tsdf_origin_xyz[1]), float(world.prior_tsdf_origin_xyz[2])]
        }, f, indent=2, ensure_ascii=False)
    print("[OK] Written:", wj)

# Visualization

def preview_plot(geoms_m: dict, out_png: str, title: str = "preview"):
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib unavailable, skipping preview")
        return
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for p in geoms_m.get(1002, []):
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.plot(x, y, linewidth=0.5, alpha=0.6)
    for p in geoms_m.get(1000, {}).get('polys', []):
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, alpha=0.2)
    for ln in geoms_m.get(888, []) + geoms_m.get(1000, {}).get('lines', []):
        if ln is None or ln.is_empty:
            continue
        x, y = ln.xy
        ax.plot(x, y, linewidth=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("[OK] Preview:", out_png)

# Main pipeline

def main():
    ap = argparse.ArgumentParser(description="OSM -> TSDF & Mesh")
    ap.add_argument('--osm', required=True, type=str, help='OSM file path')
    ap.add_argument('--outdir', required=True, type=str, help='Output directory')
    ap.add_argument('--stype-key', default='sType', type=str, help='OSM tag key for type classification, default sType')
    ap.add_argument('--voxel', default=0.10, type=float, help='Voxel resolution (meters)')
    ap.add_argument('--height', default=3.0, type=float, help='Height (meters), thickness for 2.5D TSDF replication along Z')
    ap.add_argument('--trunc', default=0.40, type=float, help='TSDF truncation distance (meters)')
    ap.add_argument('--scale', default=None, type=float, help='Manual scale (m/OSM unit); if given, skip slot-based estimation')
    ap.add_argument('--slot-width', default=2.5, type=float, help='Target parking slot width (meters) for scale estimation')
    ap.add_argument('--no-preview', action='store_true', help='Skip preview image export')

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[1/6] Parsing OSM ...")
    nodes, ways = load_osm(args.osm, args.stype_key)
    geoms_raw = build_geometries(nodes, ways)

    if len(nodes) == 0:
        raise RuntimeError("OSM has no nodes")
    arr = np.array(list(nodes.values()), dtype=np.float64)
    centroid_osm = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    print("[2/6] Estimating dominant orientation and rotating to axis-aligned ...")
    rot_apply_deg = dominant_angle_deg(geoms_raw)

    print("   Rotation angle (deg):", rot_apply_deg)
    if args.scale is None:
        print("[3/6] Estimating scale S (m/OSM) from slots ...")
        S = estimate_scale_from_slots(geoms_raw.get(1002, []), target_width_m=args.slot_width)
        if S is None:
            print("   [WARN] Insufficient slot samples, using S=1.0. Use --scale to specify.")
            S = 1.0
    else:
        S = float(args.scale)
    print("   Scale S:", S)

    print("[4/6] Applying rotation and scale, converting to metric ...")
    geoms_m = apply_rotation_and_scale(geoms_raw, centroid_osm, rot_apply_deg, S)

    print("[5/6] Rasterizing and computing TSDF ...")
    mask, origin_xy0, dims2 = rasterize_geometry(geoms_m, voxel=args.voxel)
    tsdf, voxel_size, dims3 = compute_tsdf_from_mask(mask, voxel=args.voxel, height=args.height, trunc=args.trunc)

    prior_tsdf_origin_xyz = (float(origin_xy0[0]), float(origin_xy0[1]), 0.0)

    tsdf_npz = os.path.join(args.outdir, 'tsdf.npz')
    np.savez_compressed(tsdf_npz,
                        tsdf=tsdf,
                        voxel=np.array(voxel_size, dtype=np.float32),
                        origin=np.array(prior_tsdf_origin_xyz, dtype=np.float32),
                        dims=np.array(dims3, dtype=np.int32),
                        trunc=np.array([args.trunc], dtype=np.float32))
    print("[OK] Saved TSDF:", tsdf_npz)

    mesh_path = os.path.join(args.outdir, 'tsdf_3d_mesh.ply')
    exported = export_ply_mesh_from_tsdf(tsdf, args.voxel, prior_tsdf_origin_xyz, mesh_path)

    world = WorldTransform(
        applied_rotation_deg=float(rot_apply_deg),
        centroid_osm=(float(centroid_osm[0]), float(centroid_osm[1])),
        scale_m_per_osm=float(S),
        prior_tsdf_origin_xyz=(float(prior_tsdf_origin_xyz[0]), float(prior_tsdf_origin_xyz[1]), float(prior_tsdf_origin_xyz[2]))
    )
    save_boundary_and_world(args.outdir, world)

    if not args.no_preview:
        preview_plot(geoms_m, os.path.join(args.outdir, 'preview_xy.png'), title='After rotation+scale (meters)')

    print("[6/6] Complete!")
    if not exported:
        print("Tip: Install scikit-image for mesh export: pip install scikit-image")


if __name__ == '__main__':
    main()
