# SlamV1 - SLAM System with Conical Mirror Vision

A robot localization and mapping system using a conical (fish-eye) mirror for 360-degree vision. The system unwraps wide-angle camera images, detects grid markers, tracks robot position, and plans paths through the environment.

## Table of Contents

- [Overview](#overview)
- [Module Architecture](#module-architecture)
- [Module Documentation](#module-documentation)
  - [plane.py - Image Transformation Core](#planepy---image-transformation-core)
  - [localisation.py - Position Tracking](#localisationpy---position-tracking)
  - [router.py - Pathfinding](#routerpy---pathfinding)
  - [analyse.py - Image Analysis](#analysepy---image-analysis)
  - [image.py - Panorama Unwrapping](#imagepy---panorama-unwrapping)
  - [communicate.py - ESP32 Communication](#communicatepy---esp32-communication)
  - [testing.py - Grid Navigation](#testingpy---grid-navigation)
  - [tools.py - Utilities](#toolspy---utilities)
  - [main.py - Entry Point](#mainpy---entry-point)

---

## Overview

The system works by:
1. Capturing images through a **conical mirror** that provides 360° field of view
2. **Unwrapping** the fisheye image into a rectangular panorama
3. **Detecting grid markers** (crosses) to build a coordinate system
4. **Tracking robot movement** using optical flow
5. **Pathfinding** through the mapped environment
6. **Sending commands** to ESP32 motor controllers

---

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│                    (Entry Point / CLI)                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ localisation │ │  testing.py  │ │ communicate  │
│     .py      │ │              │ │     .py     │
└──────┬───────┘ └──────┬───────┘ └──────────────┘
       │                │
       │                ▼
       │        ┌──────────────┐
       │        │    router    │
       │        │     .py     │
       │        └──────┬───────┘
       │               │
       ▼               ▼
┌──────────────────────────────────────┐
│              plane.py                 │
│    (Core Image Transformation)        │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             image.py                  │
│   (Panorama Unwrapping & Sectors)     │
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│            analyse.py                 │
│        (Image Analysis)               │
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│            tools.py                   │
│          (Utilities)                   │
└──────────────────────────────────────┘
```

---

## Module Documentation

---

### plane.py - Image Transformation Core

**Purpose:** Core module for transforming wide-angle conical mirror images into top-down square views. This is the foundation for all image processing in the system.

**API:**

#### Functions

##### `build_combined_maps(top_size, source_width, source_height, cx, cy, outer_r, rotation_deg, field_scale, lens_deg, cone_power) -> tuple[np.ndarray, np.ndarray]`
Builds optimized remap coordinate tables for one-pass transformation from source image to top-view.

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_size` | int | Output square size in pixels |
| `source_width` | int | Source image width |
| `source_height` | int | Source image height |
| `cx` | float | Mirror center X coordinate |
| `cy` | float | Mirror center Y coordinate |
| `outer_r` | float | Outer mirror radius |
| `rotation_deg` | float | Top-view rotation in degrees |
| `field_scale` | float | Field scale (0.0-1.0) |
| `lens_deg` | float | Wide-angle correction in degrees |
| `cone_power` | float | Cone radial correction power |

**Returns:** Tuple of (map_x, map_y) remap coordinate arrays.

---

##### `remap_frame(frame, map_x, map_y, background_rgb, interpolation=cv2.INTER_LINEAR) -> np.ndarray`
Applies pre-computed remap transformation to a single frame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | np.ndarray | Input image (BGR) |
| `map_x` | np.ndarray | X coordinate map from build_combined_maps |
| `map_y` | np.ndarray | Y coordinate map from build_combined_maps |
| `background_rgb` | tuple | Background color (R, G, B) |
| `interpolation` | int | OpenCV interpolation flag |

**Returns:** Transformed top-view image.

---

##### `unwrap_image(input_path, cx, cy, outer_r, lens_deg, cone_power, rotation_deg, top_size, field_scale, output_dir, background, save_lens_corrected, use_opencv, cubic) -> Path`
Unwraps a single image or video frame to top-down view.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str/Path | required | Source image path |
| `cx` | float | DEFAULTS["cx"] | Mirror center X |
| `cy` | float | DEFAULTS["cy"] | Mirror center Y |
| `outer_r` | float | DEFAULTS["outer_r"] | Outer mirror radius |
| `lens_deg` | float | DEFAULTS["lens_deg"] | Wide-angle correction |
| `cone_power` | float | DEFAULTS["cone_power"] | Cone radial correction |
| `rotation_deg` | float | DEFAULTS["rotation_deg"] | Rotation offset |
| `top_size` | int | DEFAULTS["top_size"] | Output size |
| `field_scale` | float | DEFAULTS["field_scale"] | Field scale |
| `output_dir` | Path | "Output" | Output directory |
| `background` | tuple | (0,0,0) | Background color |
| `save_lens_corrected` | bool | False | Save intermediate lens-corrected |
| `use_opencv` | bool | True | Use OpenCV path vs numpy |
| `cubic` | bool | False | Use cubic interpolation |

---

##### `debug_parameters(input_path, cx, cy, outer_r, lens_deg, cone_power, rotation_deg, top_size, field_scale, background) -> None`
Interactive parameter tuning using matplotlib sliders.

---

##### `run_video(args) -> None`
Processes video file frame-by-frame to top-down view.

---

#### Constants

```python
DEFAULTS = {
    "cx": 1203,
    "cy": 457.0,
    "outer_r": 412.0,
    "rotation_deg": -2.0,
    "top_size": 900,
    "field_scale": 0.70,
    "lens_deg": -81.86,
    "cone_power": 2.245,
}
```

**CLI Usage:**
```bash
python plane.py -i input.jpg -o output.jpg --cx 575 --cy 457 --outer-r 412
```

---

### localisation.py - Position Tracking

**Purpose:** Tracks robot position and rotation using optical flow on unwrapped frames. Maintains cumulative displacement from starting position.

**API:**

#### Class: `ConicalLocalization`

##### `__init__(cx, cy, outer_r, lens_deg, cone_power, rotation_deg, top_size, field_scale, roi, debug_mode, background, interpolation, roi_margin, edge_margin, min_features)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cx` | float | DEFAULTS["cx"] | Mirror center X |
| `cy` | float | DEFAULTS["cy"] | Mirror center Y |
| `outer_r` | float | DEFAULTS["outer_r"] | Outer mirror radius |
| `lens_deg` | float | DEFAULTS["lens_deg"] | Wide-angle correction |
| `cone_power` | float | DEFAULTS["cone_power"] | Cone power |
| `rotation_deg` | float | DEFAULTS["rotation_deg"] | Rotation offset |
| `top_size` | int | DEFAULTS["top_size"] | Output size |
| `field_scale` | float | DEFAULTS["field_scale"] | Field scale |
| `roi` | str/Path/np.ndarray | None | ROI mask path or array |
| `debug_mode` | bool | True | Show debug overlay |
| `background` | tuple | (0,0,0) | Background color |
| `roi_margin` | int | 10 | Margin from ROI boundary |
| `edge_margin` | int | 15 | Margin from image edges |
| `min_features` | int | 10 | Min features before redetection |

---

##### `unwrap_frame(frame) -> np.ndarray`
Transforms a raw frame to top-down view.

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | np.ndarray | Input camera frame |

**Returns:** Unwrapped top-view image.

---

##### `track_displacement(unwrapped_frame) -> tuple[float, float, float, np.ndarray]`
Tracks cumulative displacement and rotation relative to starting position.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unwrapped_frame` | np.ndarray | Unwrapped top-view image |

**Returns:** Tuple of (total_x, total_y, rotation_deg, debug_overlay)

---

##### `maybe_update_features(current_time) -> bool`
Re-detects features if count drops below `min_features` or every 5 seconds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_time` | float | Current timestamp (from time.time()) |

**Returns:** True if features were updated.

---

##### `reset() -> None`
Resets all tracking state to initial values.

---

#### Example Usage

```python
from localisation import ConicalLocalization
import cv2

locator = ConicalLocalization(
    cx=308, cy=234, outer_r=230,
    lens_deg=-83.0, cone_power=2.1,
    rotation_deg=0.0, top_size=400,
    field_scale=0.70, roi="Images/1.png",
    debug_mode=True
)

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    unwrapped = locator.unwrap_frame(frame)
    x, y, rotation, debug = locator.track_displacement(unwrapped)
    print(f"Position: X={x:.1f}, Y={y:.1f}, Rotation={rotation:.1f}°")
    cv2.imshow('Debug', debug)
    if cv2.waitKey(30) == 27:
        break
```

---

### buildGraph.py - Graph Construction

**Purpose:** Builds a graph of the explored field based on sector analysis. Determines which sectors can be connected based on floor level, colored pipes, and ramps.

**API:**

#### Classes

##### `FloorLevel` (Enum)
Floor level detection values.
- `WHITE` - Light floor
- `BLACK` - Dark floor  
- `UNKNOWN` - Undetermined

##### `Pipe`
Represents a colored pipe obstacle.
- `sector`: Tuple[int, int] - Sector coordinates (row, col)
- `color`: str - "blue", "red", or "green"
- `position`: Tuple[float, float] - Pixel position

##### `Ramp`
Represents a ramp between floor levels.
- `sector`: Tuple[int, int] - Sector coordinates
- `direction_from`: str - Entry direction ('U', 'D', 'L', 'R')
- `target_level`: str - Target level ("white" or "black")
- `circles`: List[Tuple[str, float, float]] - Circle positions

##### `Sector`
Represents a grid sector.
- `row`, `col`: int - Grid coordinates
- `floor_level`: FloorLevel - Floor color
- `has_green_pipe`: bool - Contains green pipe
- `pipes`: List[Pipe] - All pipes in sector
- `ramp`: Optional[Ramp] - Ramp if present
- `visited`: bool - Has been explored

##### `BuildGraph`
Main graph construction class.

###### Methods

`__init__(grid_rows=4, grid_cols=4)`
Creates empty graph with specified dimensions.

`add_sector_analysis(row, col, analysis)`
Adds analysis result for a sector from `analyse.analyze_image()`.

`can_build_edge(from_pos, direction) -> Tuple[bool, str]`
Checks if edge can be built to adjacent sector.

`build_edges()`
Builds all possible edges based on analyzed sectors.

`get_pipe_positions(color) -> List[Pipe]`
Gets pipes of specified color ('green', 'blue', 'red').

`get_unvisited_sectors() -> List[Tuple[int, int]]`
Gets unexplored sectors with known floor level.

`is_field_explored(required_green=3, required_colored=3) -> bool`
Checks if exploration is complete (3 green + 3 colored pipes).

`get_nearest_unvisited(start) -> Optional[Tuple[int, int]]`
Finds nearest unvisited sector.

`find_path_to(start, end) -> Tuple[int, str, List[Tuple]]`
Finds path using Dijkstra's algorithm with turn minimization.

`get_status() -> dict`
Returns exploration statistics.

---

#### Edge Building Rules

An edge can be built between sectors if:
1. **Same floor level** + **no colored pipes** → Edge allowed
2. **Ramp present** (dark shape with red/blue circles):
   - Red left + Blue right + current level WHITE → Can go to BLACK
   - Blue left + Red right + current level BLACK → Can go to WHITE

---

#### Example Usage

```python
from buildGraph import BuildGraph

graph = BuildGraph(grid_rows=4, grid_cols=4)

# Add sector analysis
analysis = analyse.analyze_image('sector1.jpg')
graph.add_sector_analysis(0, 0, analysis)

# Build graph
graph.build_edges()

# Check if explored
if graph.is_field_explored():
    print("Field explored!")

# Find path
dist, commands, path = graph.find_path_to((0, 0), (2, 3))
print(f"Path: {commands}")
```

---

### router.py - Pathfinding

**Purpose:** Creates an NxN grid graph and finds optimal paths using Dijkstra's algorithm with turn minimization.

**API:**

#### Class: `Pathfinder`

##### `__init__(n: int)`
Creates a new pathfinder with n×n grid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | int | Grid dimension (n×n nodes) |

---

##### `addConnection(p1, p2, oneWay=False) -> None`
Adds an edge between two grid points.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p1` | tuple | First point (row, col) |
| `p2` | tuple | Second point (row, col) |
| `oneWay` | bool | If True, edge is directed only from p1→p2 |

---

##### `getRoute(start, end, start_direction='U') -> tuple[int, str, list[tuple]]`
Finds optimal path with minimized turns.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | tuple | required | Start position (row, col) |
| `end` | tuple | required | End position (row, col) |
| `start_direction` | str | 'U' | Initial robot direction: 'U'(up), 'R'(right), 'D'(down), 'L'(left) |

**Returns:** Tuple of (distance, command_string, path_coordinates)
- `distance`: Path length in edges
- `command_string`: Movement commands (e.g., "F3R1F2A1F1")
  - `F<n>`: Move forward n steps
  - `R`: Turn right 90°
  - `L`: Turn left 90°
  - `A`: Turn around 180°
- `path_coordinates`: List of (row, col) tuples

---

##### `visualize(path=None) -> None`
Displays graph and optional path using OpenCV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | list | Optional path to highlight |

---

#### Example Usage

```python
from router import Pathfinder

pf = Pathfinder(16)

# Add horizontal connections
for r in range(16):
    for c in range(15):
        pf.addConnection((r, c), (r, c+1))

# Add vertical connections
for c in range(16):
    for r in range(15):
        pf.addConnection((r, c), (r+1, c))

# Find path
distance, commands, path = pf.getRoute((0, 0), (15, 15), start_direction='R')
print(f"Distance: {distance}")
print(f"Commands: {commands}")
pf.visualize(path)
```

---

### analyse.py - Image Analysis

**Purpose:** Analyzes unwrapped images to detect colored objects and shapes (black floor/lines, blue/red shapes, green areas).

**API:**

#### Functions

##### `analyze_image(image_path) -> dict | None`
Analyzes a single image for colored regions and shapes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | str | Path to the image file |

**Returns:** Dictionary with detection results:

```python
{
    "black_status": str,        # "пол (много)", "линия", "мало/нет"
    "black_area_px": int,        # Pixel count of black areas
    
    "has_blue_oval": bool,       # Blue oval detected
    "has_blue_rectangle": bool,  # Blue rectangle detected
    "blue_area_px": int,         # Blue pixel count
    
    "has_red_oval": bool,       # Red oval detected
    "has_red_rectangle": bool,  # Red rectangle detected
    "has_red_line": bool,       # Red line detected
    "red_area_px": int,         # Red pixel count
    
    "has_green": bool,          # Green area present
    "green_area_px": int,       # Green pixel count
}
```

Returns `None` if image cannot be loaded.

---

##### `analyze_directory(output_dir='Output', save_masks=True) -> dict`
Analyzes all numbered images (1.jpg, 2.jpg, etc.) in a directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | 'Output' | Directory containing images |
| `save_masks` | bool | True | Save green mask images |

**Returns:** Dictionary mapping filenames to analysis results.

---

#### Color Detection Parameters

```python
BLACK_AREA_FLOOR_THRESHOLD = 10000   # Pixels for "floor" status
BLACK_AREA_MIN_THRESHOLD = 10000      # Minimum pixels
GREEN_AREA_THRESHOLD = 500            # Minimum green pixels
```

Color ranges (HSV):
- **Black**: Grayscale < 80
- **Blue**: H: 90-130, S: 40-255, V: 40-255
- **Red**: H: 0-10 or 160-180, S: 40-255, V: 40-255
- **Green**: H: 50-100, S: 130-255, V: 120-255

---

### image.py - Panorama Unwrapping

**Purpose:** Unwraps conical mirror images into panoramic view and splits into sectors for sector-based processing.

**API:**

#### Functions

##### `select_center_manually(image) -> tuple | None`
Opens interactive window for manual center selection. Left-click to select.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | np.ndarray | Input image for display |

**Returns:** Center coordinates (x, y) or None if cancelled.

---

##### `find_mirror_center_and_radius(image, manual_center=None, manual_radius=None) -> tuple`
Finds mirror center and radius using Hough circle detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | np.ndarray | Input image |
| `manual_center` | tuple | Optional manual center (x, y) |
| `manual_radius` | int | Optional manual radius |

**Returns:** Tuple of (center, radius)

---

##### `unwrap_cone_image(image_path, output_path, manual_center=None, manual_radius=None, output_width=1500, debug=False) -> np.ndarray`
Unwraps conical image to panorama.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | required | Source image path |
| `output_path` | str | required | Output panorama path |
| `manual_center` | tuple | None | Override center |
| `manual_radius` | int | None | Override radius |
| `output_width` | int | 1500 | Output panorama width |
| `debug` | bool | False | Show debug windows |

**Returns:** Unwrapped image array.

---

##### `select_and_extract_parts(unwrapped_img, output_dir, DEBUG) -> None`
Interactive tool to split panorama into 4 sectors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unwrapped_img` | np.ndarray | Panorama image |
| `output_dir` | str | Directory for sector images |
| `DEBUG` | bool | Enable interactive selection |

Saves sectors as `1.jpg`, `2.jpg`, `3.jpg`, `4.jpg`.

---

### communicate.py - ESP32 Communication

**Purpose:** Handles serial communication with ESP32 microcontroller for motor control. Implements a request-response protocol where each command expects a response from ESP.

**API:**

#### Class: `ESPCommunication`

##### `__init__(port='/dev/serial0', baud=115200, debug=False)`
Initializes serial connection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | str | '/dev/serial0' | Serial port device |
| `baud` | int | 115200 | Baud rate |
| `debug` | bool | False | Print debug messages |

Falls back to virtual mode if serial unavailable.

---

##### `sendMotionCommand(speeds, servos) -> Tuple[bool, dict]`
Sends motion command to ESP32 and waits for response.

| Parameter | Type | Description |
|-----------|------|-------------|
| `speeds` | list[4] | List of 4 int16 speed values (-32768 to 32767) |
| `servos` | list[4] | List of 4 uint16 servo positions (0 to 65535) |

**Returns:** Tuple of (success, response_data)
- `success`: True if ESP responded within timeout
- `response_data`: Dict with mode and values if successful

---

##### `sendMode(mode) -> Tuple[bool, dict]`
Sends mode change command to ESP32.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | int | Mode number (0-255) |

**Returns:** Tuple of (success, response_data)

---

##### `get_stats() -> dict`
Returns communication statistics.

**Returns:**
```python
{
    'sent': int,           # Total commands sent
    'received': int,        # Total responses received
    'missed': int,         # Missed responses count
    'success_rate': float, # Success rate percentage
    'virtual': bool        # Virtual mode active
}
```

---

##### `is_connected() -> bool`
Checks if connection is still alive.

---

##### `close() -> None`
Closes serial connection and prints final statistics.

---

#### Communication Protocol

**RPi → ESP32 (17 bytes):**
```
Header: 'M' (1 byte)
Speeds: 4 × int16 (8 bytes) - motors[0..3]
Servos: 4 × uint16 (8 bytes) - servos[0..3]
```

**ESP32 → RPi Data (18 bytes):**
```
Header: 'M' (1 byte)
Mode: uint8 (1 byte)
Values: 4 × int32 (16 bytes) - sensor values
```

**ESP32 → RPi Text (2+n bytes):**
```
Header: 'T' (1 byte)
Length: uint8 (1 byte)
Data: n bytes (UTF-8 text)
```

#### Timeout Behavior

- **Response timeout:** 500ms
- **Max missed responses:** 10 (exiting after this limit)
- **Warning logged:** "ESP did not respond to: {command}" after each timeout
- **Error logged and exit:** After MAX_MISSED_RESPONSES missed responses

---

#### Example Usage

```python
from сommunicate import ESPCommunication

esp = ESPCommunication(port='/dev/ttyUSB0', debug=True)

# Send motion command and wait for response
success, response = esp.sendMotionCommand([10, -4, 0, 0], [0, 0, 0, 0])
if success:
    print(f"Mode: {response['mode']}, Values: {response['values']}")
else:
    print("ESP did not respond")

# Check connection status
if not esp.is_connected():
    print("Connection lost!")

esp.close()
```

---

### testing.py - Grid Navigation

**Purpose:** Cross detection, grid building, and target navigation using the unwrapped top-view images.

**API:**

#### Classes

##### `TrackedCross`
Represents a tracked cross marker with velocity prediction.

###### Methods
- `predict() -> tuple`: Returns predicted (x, y) position
- `update(x, y) -> None`: Updates position with new detection
- `mark_lost() -> None`: Increments lost frame counter

---

##### `CrossTracker`
Tracks multiple crosses across frames using prediction matching.

###### Methods
- `update(detections) -> list`: Updates tracks with new detections
  - `detections`: List of (x, y) tuples
  - **Returns:** List of (x, y, age) tuples for confirmed tracks

---

#### Functions

##### `cluster_and_average(points, radius=15) -> list[tuple]`
Clusters nearby points and returns centroids.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | list | required | List of (x, y) points |
| `radius` | int | 15 | Clustering radius |

---

##### `build_grid_from_points(points, img_shape) -> tuple`
Clusters detected crosses into grid lines.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | list | Detected cross coordinates |
| `img_shape` | tuple | Image shape for reference |

**Returns:** Tuple of (grid_nodes, line_x_coords, line_y_coords)

---

##### `find_target_node(grid_nodes, robot_center, exclude_radius=30) -> tuple | None`
Finds nearest grid node to navigate toward.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_nodes` | list | required | Available grid nodes |
| `robot_center` | tuple | required | Robot position (x, y) |
| `exclude_radius` | int | 30 | Radius to exclude near robot |

**Returns:** Target node coordinates or None.

---

##### `calculate_navigation_vector(robot_pos, target_pos) -> tuple`
Calculates direction and distance to target.

| Parameter | Type | Description |
|-----------|------|-------------|
| `robot_pos` | tuple | Robot (x, y) |
| `target_pos` | tuple | Target (x, y) |

**Returns:** Tuple of (dx, dy, angle_deg)

---

##### `detect_raw_crosses(unwrapped_img, roi_mask) -> tuple`
Detects cross markers using Hough line detection and intersection finding.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unwrapped_img` | np.ndarray | Unwrapped top-view image |
| `roi_mask` | np.ndarray | Optional ROI mask |

**Returns:** Tuple of (cross_points, edges_image)

---

##### `process_frame(frame, map_x, map_y, tracker, roi_mask, bg) -> tuple`
Processes single frame for cross detection and navigation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | np.ndarray | Raw camera frame |
| `map_x` | np.ndarray | X remap coordinates |
| `map_y` | np.ndarray | Y remap coordinates |
| `tracker` | CrossTracker | Tracker instance |
| `roi_mask` | np.ndarray | Optional ROI mask |
| `bg` | tuple | Background color |

**Returns:** Tuple of (result_image, edges_view, nav_data)

---

### tools.py - Utilities

**Purpose:** Shared utility functions used across modules.

**API:**

#### Functions

##### `drawImageOnScreen(window_name, image) -> None`
Displays image scaled to max dimension of 1000px.

| Parameter | Type | Description |
|-----------|------|-------------|
| `window_name` | str | OpenCV window name |
| `image` | np.ndarray | Image to display |

---

##### `constrain(val, minn, maxx) -> int`
Constrains value between min and max (C++-style).

| Parameter | Type | Description |
|-----------|------|-------------|
| `val` | int | Input value |
| `minn` | int | Minimum bound |
| `maxx` | int | Maximum bound |

**Returns:** Constrained value.

---

### main.py - Entry Point

**Purpose:** Simple entry point demonstrating system initialization and ESP communication.

**API:**

#### Functions

##### `main() -> None`
Initializes logging and ESP communication, sends test commands.

**Actions:**
1. Configures timestamped file logging to `Output/Logs/`
2. Creates ESPCommunication instance
3. Sends mode 3 command
4. Sends test motion command: speeds=[10, -4, 0, 0], servos=[0, 0, 0, 0]
5. Closes connection

---

## Data Flow Example

```
Camera Frame
    │
    ▼
┌─────────────────────┐
│   plane.py          │
│  (unwrap_frame)     │
│                     │
│  map_x, map_y       │
│  from build_        │
│  combined_maps()    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Unwrapped Top-View │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────────┐
│testing. │  │localisation │
│   py    │  │    .py      │
└────┬────┘  └──────┬──────┘
     │              │
     ▼              ▼
┌─────────┐  ┌─────────────┐
│ Grid &  │  │  Position  │
│ Target  │  │  X, Y, Rot │
└────┬────┘  └──────┬──────┘
     │              │
     └──────┬──────┘
            ▼
     ┌─────────────┐
     │  router.py  │
     │ (getRoute)  │
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │communicate. │
     │    py       │
     │(sendMotion) │
     └─────────────┘
```

## Dependencies

- **opencv-contrib-python** >= 4.13.0: Image processing and video I/O
- **numpy** >= 2.5.0: Numerical operations
- **pyserial** >= 3.5: Serial communication
- **matplotlib** >= 3.11.0: Debug visualization (optional)

## File Structure

```
SlamV1/
├── main.py           # Entry point
├── plane.py          # Core image transformation
├── localisation.py    # Position tracking
├── router.py         # Pathfinding
├── analyse.py        # Image analysis
├── image.py          # Panorama unwrapping
├── communicate.py    # ESP32 communication
├── testing.py        # Grid navigation
├── tools.py          # Utilities
├── сomminucate.ino   # Arduino/ESP32 firmware
├── Images/           # Test images and videos
├── Output/           # Processing outputs
│   ├── Logs/         # Application logs
│   └── green_masks/  # Generated masks
├── requirements.txt  # Python dependencies
└── LICENSE
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic localization
python localisation.py

# Run grid navigation testing
python testing.py

# Unwrap an image
python image.py

# Analyze unwrapped images
python analyse.py

# Test ESP communication
python main.py
```