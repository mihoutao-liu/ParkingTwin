#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OSM Pose Selector - Interactive tool for selecting camera poses on OSM map

Updates:
- Integrated map rotation logic from draw_osm.py
- Only calculates rotation based on 2-node ways (walls/lines)
- Applies rotation only if dominant angle > 1.0 degree
"""

import os
import sys
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from shapely.geometry import Point, LineString, Polygon
import yaml
import math
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    print("Warning: pyperclip not installed. Install with 'pip install pyperclip'.")

DEFAULT_OSM_PATH = "Datasets/ICPARKOSM/ICPARK.osm"
DEFAULT_CONFIG_PATH = "configs/config_default.yaml"
OUTPUT_DIR = "output/pose_selector"

OSM_STYPE_TAG_KEY = "sType"
CLOSURE_TOLERANCE_OSM_UNITS = 1e-5
CAR_SPACE_WIDTH = 2.5  # meters
COLOR_WALL_PILLAR = "gray"
COLOR_PARKING_BG = "lightblue"
COLOR_POSE_ARROW = "red"

# Global variables
start_point = None
current_point = None
arrow = None
arrow_head = None
pose_text = None
scale_factor = 1.0
map_rotation_angle = 0.0
fig = None
ax = None
snap_to_grid = False
is_dragging = False

# ---------------------------------------------------------------------
# Helper Functions (Geometry & Parsing)
# ---------------------------------------------------------------------

def plot_geometry(ax, geom, color, linewidth, alpha=0.7, fill=True, zorder=1):
    if geom is None or geom.is_empty: 
        return
    
    if isinstance(geom, (Polygon,)):
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=zorder)
        if fill: 
            ax.fill(x, y, alpha=alpha*0.5, fc=color, ec='none', zorder=zorder-0.5 if zorder > 0 else 0)
        
    elif isinstance(geom, (LineString,)):
        x, y = geom.xy
        ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=zorder)

def _parse_raw_osm_data(file_path: str) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OSM file not found: {file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse OSM file: {e}")

    nodes_coords = {}
    for node_el in root.findall("node"):
        try:
            nodes_coords[node_el.attrib["id"]] = (float(node_el.attrib["lon"]), float(node_el.attrib["lat"]))
        except ValueError:
            continue
    
    raw_ways = []
    for way_el in root.findall("way"):
        raw_ways.append({
            "id": way_el.attrib["id"],
            "nds": [nd.attrib["ref"] for nd in way_el.findall("nd")],
            "tags": {tag.attrib["k"]: tag.attrib["v"] for tag in way_el.findall("tag")}
        })
    return nodes_coords, raw_ways

# ---------------------------------------------------------------------
# Rotation Logic (Imported from draw_osm.py)
# ---------------------------------------------------------------------

def _get_angle(p1, p2):
    """Calculate angle between two points (0-360 degrees)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dy, dx))
    return ang if ang >= 0 else ang + 360

def _calculate_map_rotation_angle(raw_nodes_coords, raw_ways):
    """
    Calculate dominant orientation based on 2-node ways.
    Logic from draw_osm.py
    """
    angles = []
    for way in raw_ways:
        refs = way["nds"]
        # Strict check: only consider ways with exactly 2 nodes (walls/lines)
        if len(refs) != 2:
            continue
            
        if refs[0] in raw_nodes_coords and refs[1] in raw_nodes_coords:
            c1 = np.array(raw_nodes_coords[refs[0]])
            c2 = np.array(raw_nodes_coords[refs[1]])
            
            if np.allclose(c1, c2):
                continue
            
            # Use _get_angle and modulo 90
            angles.append(_get_angle(c1, c2) % 90)

    if not angles:
        return 0.0
    return float(np.mean(angles))

def _rotate_point(point, angle_degrees, center=(0.0, 0.0)):
    """Rotate a single point"""
    angle_rad = math.radians(angle_degrees)
    ox, oy = center
    px, py = point
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return qx, qy

def _rotate_all_nodes(nodes_coords_dict, rotation_angle_deg):
    """Rotate all nodes around their centroid"""
    rotated_nodes = {}
    all_coords = np.array(list(nodes_coords_dict.values()))
    
    if all_coords.size > 0:
        centroid = tuple(np.mean(all_coords, axis=0)) 
    else:
        centroid = (0.0, 0.0) 

    for nid, coord in nodes_coords_dict.items():
        rotated_nodes[nid] = _rotate_point(coord, rotation_angle_deg, center=centroid)
    return rotated_nodes

# ---------------------------------------------------------------------
# Map Processing
# ---------------------------------------------------------------------

def _create_shapely_elements_from_osm_data(processed_nodes_coords, raw_ways, s_type_tag_key, closure_tolerance):
    elements = []
    for way_info in raw_ways:
        way_id = way_info["id"]
        node_refs = way_info["nds"]
        tags = way_info["tags"]

        s_type_str_val = tags.get(s_type_tag_key)
        if s_type_str_val not in ("888", "1000", "1002"):
            continue
        s_type = s_type_str_val

        coords = []
        valid_refs = True
        for ref in node_refs:
            if ref in processed_nodes_coords:
                coords.append(processed_nodes_coords[ref])
            else:
                valid_refs = False
                break
        
        if not valid_refs or len(coords) < 2: continue

        shape = None
        is_closed = len(node_refs) > 2 and node_refs[0] == node_refs[-1]
        
        # Check tolerance closure
        if not is_closed and len(coords) >= 3:
            p_start = Point(coords[0])
            p_end = Point(coords[-1])
            if p_start.distance(p_end) < closure_tolerance: 
                is_closed = True
                if p_start.distance(p_end) > 1e-9: 
                    coords.append(coords[0])
        
        try:
            if is_closed:
                unique_points = set(map(tuple, coords))
                if len(unique_points) < 3:
                    shape = LineString(coords)
                else:
                    poly = Polygon(coords)
                    if poly.is_valid:
                        shape = poly
                    else:
                        shape = LineString(coords)
            else:
                shape = LineString(coords)
        except Exception: 
            shape = None

        if shape and not shape.is_empty:
            elements.append({"id": way_id, "sType": s_type, "geometry": shape, "tags": tags})
            
    return elements

def estimate_scale(slots_osm_shapely, car_space_width=None):
    if car_space_width is None:
        car_space_width = CAR_SPACE_WIDTH
        
    if not slots_osm_shapely:
        print("Warning: No parking slots found, using scale=1.0")
        return 1.0
    
    short_sides = []
    for poly in slots_osm_shapely:
        if poly.exterior is None: continue
        coords = list(poly.exterior.coords)
        if len(coords) < 4: continue
        
        s1 = Point(coords[0]).distance(Point(coords[1]))
        s2 = Point(coords[1]).distance(Point(coords[2]))
        cur = min(s1, s2)
        if cur > 1e-6: 
            short_sides.append(cur)
    
    if short_sides:
        mean_len = float(np.mean(short_sides))
        print(f"Avg slot short side in OSM units: {mean_len:.4f}")
        if mean_len > 1e-9: 
            return car_space_width / mean_len
    
    print("Warning: cannot estimate scale from slots, using 1.0")
    return 1.0

def project_coords_to_meters(coords_list_tuples):
    return [(lon * scale_factor, lat * scale_factor) for lon, lat in coords_list_tuples]

def load_osm_map(osm_path):
    global scale_factor, map_rotation_angle

    print(f"[OSM] loading: {osm_path}")
    raw_nodes, raw_ways = _parse_raw_osm_data(osm_path)
    print(f"  nodes: {len(raw_nodes)}, ways: {len(raw_ways)}")

    # --- Rotation Logic from draw_osm.py ---
    dominant_angle_0_90 = _calculate_map_rotation_angle(raw_nodes, raw_ways)
    
    # Only rotate if the angle is significant (> 1.0 degree)
    if abs(dominant_angle_0_90) > 1.0:
        rotation_to_apply_deg = -dominant_angle_0_90
        nodes_rot = _rotate_all_nodes(raw_nodes, rotation_to_apply_deg)
        map_rotation_angle = dominant_angle_0_90
        print(f"  rotated map by {rotation_to_apply_deg:.2f} deg to align main direction")
    else:
        nodes_rot = raw_nodes
        map_rotation_angle = 0.0
        print("  rotation skipped (dominant direction ~ 0 deg)")
    # ---------------------------------------

    elements = _create_shapely_elements_from_osm_data(
        nodes_rot, raw_ways, OSM_STYPE_TAG_KEY, CLOSURE_TOLERANCE_OSM_UNITS
    )

    walls_osm = []
    polys_osm = []
    slots_osm = []
    
    for el in elements:
        st = el["sType"]
        geom = el["geometry"]
        if st == "888":
            if isinstance(geom, LineString): walls_osm.append(geom)
        elif st == "1000":
            if isinstance(geom, Polygon): polys_osm.append(geom)
            elif isinstance(geom, LineString): walls_osm.append(geom)
        elif st == "1002":
            if isinstance(geom, Polygon): slots_osm.append(geom)

    scale_factor = estimate_scale(slots_osm)
    print(f"  scale: {scale_factor:.4f} m / OSM-unit")

    walls_m = [LineString(project_coords_to_meters(list(line.coords))) for line in walls_osm if not line.is_empty]
    
    closed_areas_m = []
    for poly in polys_osm:
        if poly.exterior is not None:
            proj = project_coords_to_meters(list(poly.exterior.coords))
            if len(proj) >= 3:
                new_poly = Polygon(proj)
                if new_poly.is_valid and not new_poly.is_empty:
                    closed_areas_m.append(new_poly)

    slots_m = []
    for poly in slots_osm:
        if poly.exterior is not None:
            proj = project_coords_to_meters(list(poly.exterior.coords))
            if len(proj) >= 3:
                new_poly = Polygon(proj)
                if new_poly.is_valid and not new_poly.is_empty:
                    slots_m.append(new_poly)

    return walls_m, closed_areas_m, slots_m

# ---------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------

def on_click(event):
    global start_point, current_point, arrow, arrow_head, pose_text, is_dragging
    
    if not event.key == 'control':
        return
        
    if event.inaxes:
        start_point = (event.xdata, event.ydata)
        current_point = start_point
        # Clear previous markers
        if arrow: arrow.remove()
        if arrow_head: arrow_head.remove()
        if pose_text: pose_text.remove()
        arrow, arrow_head, pose_text = None, None, None
        
        arrow = plt.plot([start_point[0]], [start_point[1]], 'ro-', markersize=8)[0]
        pose_text = plt.text(start_point[0], start_point[1] + 1, 
                           f"Pos: ({start_point[0]:.1f}, {start_point[1]:.1f})\nOri: 0°", 
                           color='red', fontsize=10, ha='center')
        plt.draw()
        is_dragging = True

def on_motion(event):
    global start_point, current_point, arrow, arrow_head, pose_text, is_dragging
    
    if not is_dragging or not event.key == 'control':
        return
        
    if event.inaxes and start_point:
        current_point = (event.xdata, event.ydata)
        dx = current_point[0] - start_point[0]
        dy = current_point[1] - start_point[1]
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        original_angle_deg = angle_deg
        
        if snap_to_grid:
            angle_deg = round(angle_deg / 90) * 90
            angle_rad = math.radians(angle_deg)
            length = math.sqrt(dx**2 + dy**2)
            dx = length * math.cos(angle_rad)
            dy = length * math.sin(angle_rad)
            current_point = (start_point[0] + dx, start_point[1] + dy)
        
        if arrow:
            arrow.set_data([start_point[0], current_point[0]], [start_point[1], current_point[1]])
        
        # Redraw Arrow Head
        if arrow_head: arrow_head.remove()
        head_len = min(0.5, math.sqrt(dx**2 + dy**2) * 0.3)
        angle1 = angle_rad + math.radians(150)
        angle2 = angle_rad - math.radians(150)
        x1 = current_point[0] + head_len * math.cos(angle1)
        y1 = current_point[1] + head_len * math.sin(angle1)
        x2 = current_point[0] + head_len * math.cos(angle2)
        y2 = current_point[1] + head_len * math.sin(angle2)
        arrow_head = plt.plot([current_point[0], x1, x2, current_point[0]], 
                             [current_point[1], y1, y2, current_point[1]], 'r-')[0]
        
        # Update Text
        if pose_text: pose_text.remove()
        real_angle = (angle_deg - map_rotation_angle) % 360
        angle_text = f"Ori: {real_angle:.1f}°"
        if snap_to_grid:
            angle_text += f" (Snap)"
        
        pose_text = plt.text(start_point[0], start_point[1] + 1, 
                           f"Pos: ({start_point[0]:.1f}, {start_point[1]:.1f})\n{angle_text}", 
                           color='red', fontsize=10, ha='center')
        plt.draw()

def on_release(event):
    global is_dragging
    if is_dragging and start_point and current_point and start_point != current_point:
        print("\nPose selection complete!")
        print_pose_matrix(start_point, current_point)
        is_dragging = False

def on_key(event):
    global start_point, current_point, arrow, arrow_head, pose_text, snap_to_grid
    
    if event.key == 'r':
        print("\nResetting pose selection")
        start_point, current_point = None, None
        if arrow: arrow.remove()
        if arrow_head: arrow_head.remove()
        if pose_text: pose_text.remove()
        arrow, arrow_head, pose_text = None, None, None
        plt.draw()
    
    elif event.key == 'q':
        print("\nExiting program")
        plt.close()
    
    elif event.key == 'g':
        snap_to_grid = not snap_to_grid
        print(f"\nGrid alignment: {'ON' if snap_to_grid else 'OFF'}")
        update_title()
        plt.draw()
        if start_point and current_point:
            # Fake a motion event to update the arrow snap immediately
            dummy_event = type('obj', (object,), {'key': 'control', 'inaxes': ax, 'xdata': current_point[0], 'ydata': current_point[1]})
            # Need to re-trigger global state logic, simplified here by next move
    
    elif event.key == 's' and start_point and current_point:
        save_pose_to_config(start_point, current_point)
    
    elif event.key == 'c' and start_point and current_point:
        if CLIPBOARD_AVAILABLE:
            pose_str = format_pose_matrix_for_clipboard(start_point, current_point)
            pyperclip.copy(pose_str)
            print("\nPose matrix copied to clipboard")

def calculate_pose_matrix(start_point, end_point):
    position = np.array([start_point[0], start_point[1], 0])
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # Adjust for map rotation to get world angle
    angle_rad_map = math.atan2(dy, dx)
    angle_rad_world = angle_rad_map - math.radians(map_rotation_angle)
    
    direction_length = math.sqrt(dx**2 + dy**2)
    dx_w = direction_length * math.cos(angle_rad_world)
    dy_w = direction_length * math.sin(angle_rad_world)
    
    # Construct R matrix (Z-up)
    forward = np.array([dx_w, dy_w, 0])
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = np.array([0, 0, 1])
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(forward, right)
    
    rotation = np.column_stack((right, up, forward))
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = position
    return pose_matrix

def print_pose_matrix(start_point, end_point):
    pose_matrix = calculate_pose_matrix(start_point, end_point)
    print("\nCalculated Pose Matrix:")
    print(np.array2string(pose_matrix, precision=6))
    print("\nFormat for ESLAM 'first_frame_abs_pose':")
    print("first_frame_abs_pose:")
    for i in range(4):
        vals = ", ".join([f"{val:.6f}" for val in pose_matrix[i]])
        suffix = ["R | t", "tx", "ty", "tz"][i] if i < 4 else "" 
        print(f"  - [{vals}]")

def format_pose_matrix_for_clipboard(start_point, end_point):
    pose_matrix = calculate_pose_matrix(start_point, end_point)
    res = "first_frame_abs_pose:\n"
    for i in range(4):
        vals = ", ".join([f"{val:.6f}" for val in pose_matrix[i]])
        res += f"  - [{vals}]\n"
    return res

def save_pose_to_config(start_point, end_point, config_path=DEFAULT_CONFIG_PATH):
    pose_matrix = calculate_pose_matrix(start_point, end_point)
    
    try:
        # Simple read/append/replace logic to preserve comments
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            lines = []

        new_lines = []
        skip = False
        inserted = False
        
        # Format the new block
        pose_block = "first_frame_abs_pose:\n"
        for i in range(4):
            vals = ", ".join([f"{val:.6f}" for val in pose_matrix[i]])
            comment = ["Rotation | Translation", "tx", "ty", "tz"][i]
            pose_block += f"  - [{vals}]     # {comment}\n"

        for line in lines:
            if "first_frame_abs_pose:" in line:
                new_lines.append(pose_block)
                skip = True
                inserted = True
            elif skip and line.strip().startswith("-"):
                continue # Skip old matrix lines
            else:
                if skip and line.strip() == "": skip = False # Stop skipping on empty line
                if not skip: new_lines.append(line)
        
        if not inserted:
            new_lines.append("\n" + pose_block)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Saved to {config_path}")
        
    except Exception as e:
        print(f"Error saving config: {e}")

def create_interactive_map(walls_m, closed_areas_m, slots_m):
    global ax, fig
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for slot in slots_m:
        plot_geometry(ax, slot, color=COLOR_PARKING_BG, linewidth=1, fill=True, alpha=0.5, zorder=1)
    for area in closed_areas_m:
        plot_geometry(ax, area, color=COLOR_WALL_PILLAR, linewidth=1.5, fill=True, alpha=0.7, zorder=2)
    for wall in walls_m:
        plot_geometry(ax, wall, color=COLOR_WALL_PILLAR, linewidth=2, fill=False, zorder=3)
    
    controls_text = "Controls:\nCtrl+Click: Set Pos\nCtrl+Drag: Set Ori\nr: Reset\nc: Copy\ns: Save\ng: Grid\nq: Quit"
    plt.figtext(0.02, 0.02, controls_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    update_title()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Legend
    le = [
        Line2D([0], [0], color=COLOR_WALL_PILLAR, lw=2, label='Walls'),
        patches.Patch(facecolor=COLOR_WALL_PILLAR, alpha=0.7, label='Pillars'),
        patches.Patch(facecolor=COLOR_PARKING_BG, alpha=0.5, label='Parking'),
        Line2D([0], [0], color=COLOR_POSE_ARROW, lw=2, marker='o', label='Pose')
    ]
    ax.legend(handles=le, loc='upper right')
    
    plt.show()

def update_title():
    grid = "ON" if snap_to_grid else "OFF"
    ax.set_title(f"OSM Pose Selector (Scale: {scale_factor:.4f}, Rot: {-map_rotation_angle:.2f}°, Grid: {grid})")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    osm_path = DEFAULT_OSM_PATH
    if len(sys.argv) > 1:
        osm_path = sys.argv[1]
    
    if not os.path.exists(osm_path):
        print(f"Error: {osm_path} not found.")
        return
    
    walls, areas, slots = load_osm_map(osm_path)
    print("\n" + "="*60)
    print("INSTRUCTIONS: Hold Ctrl to Click (Position) and Drag (Orientation)")
    print("="*60)
    create_interactive_map(walls, areas, slots)

if __name__ == "__main__":
    main()