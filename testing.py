import cv2
import numpy as np
from plane import remap_frame, build_combined_maps
import math
import os

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
REMAPPING_PARAMS = dict(
    cx=308, cy=234, outer_r=230,
    lens_deg=-81.86, cone_power=2.245,
    rotation_deg=-2.0, top_size=640, field_scale=0.70,
    background=(0, 0, 0),
)

ROI_PATH = 'Images/1.png'

# Детекция
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 15
ANGLE_TOLERANCE = 10 

# Кластеризация и Сетка
CLUSTER_RADIUS = 15         
GRID_CLUSTER_EPS = 20.0     
MIN_POINTS_PER_LINE = 3     

# Трекинг
MATCH_DISTANCE = 30
MIN_TRACK_AGE = 2
MAX_LOST_FRAMES = 5
PREDICTION_ALPHA = 0.7

# ==============================================================================
# ТРЕКИНГ
# ==============================================================================
class TrackedCross:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.age = 0
        self.visible_count = 0
        self.lost_frames = 0

    def predict(self):
        return self.x + self.vx, self.y + self.vy

    def update(self, x, y):
        dx, dy = x - self.x, y - self.y
        self.vx = PREDICTION_ALPHA * self.vx + (1 - PREDICTION_ALPHA) * dx
        self.vy = PREDICTION_ALPHA * self.vy + (1 - PREDICTION_ALPHA) * dy
        self.x, self.y = float(x), float(y)
        self.age += 1
        self.visible_count += 1
        self.lost_frames = 0

    def mark_lost(self):
        self.lost_frames += 1
        self.age += 1

class CrossTracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        matched = set()
        for track in self.tracks:
            px, py = track.predict()
            best_dist, best_idx = float('inf'), -1
            for i, (dx, dy) in enumerate(detections):
                if i in matched: continue
                d = math.hypot(dx - px, dy - py)
                if d < MATCH_DISTANCE and d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx != -1:
                track.update(detections[best_idx][0], detections[best_idx][1])
                matched.add(best_idx)
            else:
                track.mark_lost()

        for i, (x, y) in enumerate(detections):
            if i not in matched:
                self.tracks.append(TrackedCross(x, y))

        self.tracks = [t for t in self.tracks if t.lost_frames <= MAX_LOST_FRAMES]
        return [(int(round(t.x)), int(round(t.y)), t.age)
                for t in self.tracks if t.visible_count >= MIN_TRACK_AGE]

def cluster_and_average(points, radius=CLUSTER_RADIUS):
    if not points: return []
    pts = [(float(p[0]), float(p[1])) for p in points]
    used = [False] * len(pts)
    averaged = []
    for i, (x, y) in enumerate(pts):
        if used[i]: continue
        cx_list, cy_list = [x], [y]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]: continue
            if math.hypot(pts[j][0] - x, pts[j][1] - y) <= radius:
                cx_list.append(pts[j][0])
                cy_list.append(pts[j][1])
                used[j] = True
        averaged.append((sum(cx_list) / len(cx_list), sum(cy_list) / len(cy_list)))
    return averaged

# ==============================================================================
# ПОСТРОЕНИЕ СЕТКИ И ЛОГИКА ЦЕЛИ
# ==============================================================================

def build_grid_from_points(points, img_shape):
    if not points:
        return [], [], []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def cluster_1d(values, eps):
        if not values: return []
        sorted_v = sorted(values)
        clusters = []
        current_cluster = [sorted_v[0]]
        
        for v in sorted_v[1:]:
            if v - current_cluster[-1] <= eps:
                current_cluster.append(v)
            else:
                if len(current_cluster) >= MIN_POINTS_PER_LINE:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [v]
        
        if len(current_cluster) >= MIN_POINTS_PER_LINE:
            clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters

    line_x_coords = cluster_1d(xs, GRID_CLUSTER_EPS) 
    line_y_coords = cluster_1d(ys, GRID_CLUSTER_EPS) 

    grid_nodes = []
    for lx in line_x_coords:
        for ly in line_y_coords:
            grid_nodes.append((lx, ly))

    return grid_nodes, line_x_coords, line_y_coords


def find_target_node(grid_nodes, robot_center, exclude_radius=30):
    if not grid_nodes:
        return None
    
    rx, ry = robot_center
    best_node = None
    min_dist = float('inf')

    for node in grid_nodes:
        dist = math.hypot(node[0] - rx, node[1] - ry)
        if dist > exclude_radius and dist < min_dist:
            min_dist = dist
            best_node = node
            
    return best_node

def calculate_navigation_vector(robot_pos, target_pos):
    if target_pos is None:
        return None, None, None
        
    rx, ry = robot_pos
    tx, ty = target_pos
    
    dx = tx - rx
    dy = ty - ry
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return dx, dy, angle_deg

# ==============================================================================
# ДЕТЕКЦИЯ С МОРФОЛОГИЕЙ
# ==============================================================================
def detect_raw_crosses(unwrapped_img, roi_mask):
    gray = cv2.cvtColor(unwrapped_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, roi_mask)

    # Морфологическое закрытие для устранения разрывов
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    crosses = []
    if lines is None:
        return crosses, edges

    h, w = unwrapped_img.shape[:2]
    horiz, vert = [], []
    
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        seg = (x1, y1, x2, y2)
        
        if ang < ANGLE_TOLERANCE or ang > 180 - ANGLE_TOLERANCE:
            horiz.append(seg)
        elif 90 - ANGLE_TOLERANCE < ang < 90 + ANGLE_TOLERANCE:
            vert.append(seg)

    for hx1, hy1, hx2, hy2 in horiz:
        for vx1, vy1, vx2, vy2 in vert:
            if min(hx1, hx2) <= vx1 <= max(hx1, hx2) and min(vy1, vy2) <= hy1 <= max(vy1, vy2):
                crosses.append((vx1, hy1))
                
    return crosses, edges

# ==============================================================================
# ОБРАБОТКА КАДРА
# ==============================================================================
def process_frame(frame, map_x, map_y, tracker, roi_mask, bg=(0, 0, 0)):
    try:
        unwrapped = remap_frame(frame, map_x, map_y, bg, cv2.INTER_LINEAR)
        if unwrapped is None or unwrapped.size == 0:
            return None, None, None

        h, w = unwrapped.shape[:2]
        center = (w // 2, h // 2)

        raw, edges = detect_raw_crosses(unwrapped, roi_mask)
        averaged_raw = cluster_and_average(raw)
        confirmed = tracker.update(averaged_raw)
        confirmed_pts = [(x, y) for x, y, _ in confirmed]
        
        grid_nodes, lines_x, lines_y = build_grid_from_points(confirmed_pts, unwrapped.shape)
        target_node = find_target_node(grid_nodes, center)
        dx, dy, angle = calculate_navigation_vector(center, target_node)

        result = unwrapped.copy()
        
        for lx in lines_x:
            cv2.line(result, (int(lx), 0), (int(lx), h), (255, 100, 0), 1)
        for ly in lines_y:
            cv2.line(result, (0, int(ly)), (w, int(ly)), (255, 100, 0), 1)
            
        for nx, ny in grid_nodes:
            cv2.circle(result, (int(nx), int(ny)), 4, (255, 255, 0), -1)

        for x, y, age in confirmed:
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

        if target_node:
            tx, ty = int(target_node[0]), int(target_node[1])
            cv2.circle(result, (tx, ty), 10, (0, 0, 255), 3)
            cv2.putText(result, "TARGET", (tx+15, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if dx is not None:
                cv2.arrowedLine(result, center, (int(center[0]+dx), int(center[1]+dy)), (0, 0, 255), 2)
                info_text = f"Dist: {math.hypot(dx,dy):.1f}, Ang: {angle:.1f}"
                cv2.putText(result, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        nav_data = {
            'target': target_node,
            'dx': dx,
            'dy': dy,
            'angle': angle,
            'grid_nodes': grid_nodes
        }

        return result, debug, nav_data

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    video_path = 'Images/vid1.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    height, width = first_frame.shape[:2]

    print("Building transformation maps...")
    map_x, map_y = build_combined_maps(
        top_size=REMAPPING_PARAMS['top_size'],
        source_width=width, source_height=height,
        cx=REMAPPING_PARAMS['cx'], cy=REMAPPING_PARAMS['cy'],
        outer_r=REMAPPING_PARAMS['outer_r'],
        rotation_deg=REMAPPING_PARAMS['rotation_deg'],
        field_scale=REMAPPING_PARAMS['field_scale'],
        lens_deg=REMAPPING_PARAMS['lens_deg'],
        cone_power=REMAPPING_PARAMS['cone_power'],
    )
    print("Maps built successfully!")

    roi_mask = None
    if os.path.exists(ROI_PATH):
        roi_raw = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
        if roi_raw is not None:
            top_size = REMAPPING_PARAMS['top_size']
            roi_mask = cv2.resize(roi_raw, (top_size, top_size), interpolation=cv2.INTER_NEAREST)
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
            print(f"ROI mask loaded from {ROI_PATH}")

    tracker = CrossTracker()

    cv2.namedWindow('Grid Navigation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edges View', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, edges_view, nav_data = process_frame(
            frame, map_x, map_y, tracker, roi_mask,
            REMAPPING_PARAMS['background']
        )

        if result_img is not None:
            cv2.imshow('Grid Navigation', result_img)
            if edges_view is not None:
                cv2.imshow('Edges View', edges_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()