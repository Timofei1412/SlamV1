#!/usr/bin/env python3
"""
Localization using conical mirror unwrapping and optical flow tracking.
Tracks X, Y displacement and rotation angle relative to starting position.
Supports optional ROI mask (black areas are ignored for feature detection).
Uses the optimized one-pass OpenCV remap from plane.py (build_combined_maps).
"""
import cv2
import numpy as np
import time
from pathlib import Path
import sys
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except:
    USE_PICAMERA = False

sys.path.append(str(Path(__file__).parent))
from plane import (build_combined_maps, remap_frame, DEFAULTS)


def load_roi_mask(roi_source, target_size: int) -> np.ndarray | None:
    """Load and prepare an ROI mask for feature detection."""
    if roi_source is None:
        return None
    if isinstance(roi_source, (str, Path)):
        mask = cv2.imread(str(roi_source), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: could not load ROI from {roi_source}, ignoring")
            return None
    elif isinstance(roi_source, np.ndarray):
        if len(roi_source.shape) == 3:
            mask = cv2.cvtColor(roi_source, cv2.COLOR_BGR2GRAY)
        else:
            mask = roi_source.copy()
    else:
        return None
    if mask.shape[0] != target_size or mask.shape[1] != target_size:
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask


class ConicalLocalization:
    def __init__(self,
            cx: float = DEFAULTS["cx"],
            cy: float = DEFAULTS["cy"],
            outer_r: float = DEFAULTS["outer_r"],
            lens_deg: float = DEFAULTS["lens_deg"],
            cone_power: float = DEFAULTS["cone_power"],
            rotation_deg: float = DEFAULTS["rotation_deg"],
            top_size: int = DEFAULTS["top_size"],
            field_scale: float = DEFAULTS["field_scale"],
            roi: str | Path | np.ndarray | None = None,
            debug_mode: bool = True,
            background: tuple[int, int, int] = (0, 0, 0),
            interpolation: int = cv2.INTER_LINEAR,
            
            # New parameters for point filtering
            roi_margin: int = 10,      # Distance from ROI boundary to filter points
            edge_margin: int = 15,     # Distance from image edges to filter points
            min_features: int = 10,):    # Minimum features before re-detection
        
        self.cx = cx
        self.cy = cy
        self.outer_r = outer_r
        self.lens_deg = lens_deg
        self.cone_power = cone_power
        self.rotation_deg = rotation_deg
        self.top_size = top_size
        self.field_scale = field_scale
        self.debug_mode = debug_mode
        self.background = background
        self.interpolation = interpolation
        
        # Point filtering parameters
        self.roi_margin = roi_margin
        self.edge_margin = edge_margin
        self.min_features = min_features
        
        # Remap maps (built lazily on first frame)
        self.combined_map_x = None
        self.combined_map_y = None
        
        # ROI mask
        self.roi_mask = load_roi_mask(roi, top_size)
        if self.roi_mask is not None:
            active = int(np.count_nonzero(self.roi_mask))
            total = self.roi_mask.shape[0] * self.roi_mask.shape[1]
            print(f"ROI mask loaded: {active}/{total} ({100*active/total:.1f}%) active")
            # Erode ROI mask by roi_margin to create safe zone
            if roi_margin > 0:
                kernel = np.ones((roi_margin*2+1, roi_margin*2+1), np.uint8)
                self.roi_safe = cv2.erode(self.roi_mask, kernel, iterations=1)
            else:
                self.roi_safe = self.roi_mask.copy()
        else:
            print("No ROI mask - all areas active")
            self.roi_safe = None
        
        # Optical flow parameters
        self.feature_params = dict(
            maxCorners=150,
            qualityLevel=0.3,
            minDistance=15,
            blockSize=12
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.07)
        )
        
        # State
        self.prev_gray = None
        self.p0 = None
        self.last_update_time = None
        self.frame_count = 0
        self.total_x = 0.0
        self.total_y = 0.0
        self.total_rotation = 0.0
        self.mask = None
    
    def _build_maps(self, frame_width: int, frame_height: int) -> None:
        """Build combined remap maps for the given source frame size."""
        self.combined_map_x, self.combined_map_y = build_combined_maps(
            top_size=self.top_size,
            source_width=frame_width,
            source_height=frame_height,
            cx=self.cx,
            cy=self.cy,
            outer_r=self.outer_r,
            rotation_deg=self.rotation_deg,
            field_scale=self.field_scale,
            lens_deg=self.lens_deg,
            cone_power=self.cone_power,
        )
        print(f"Remap maps built: source={frame_width}x{frame_height}, top={self.top_size}x{self.top_size}")
    
    def unwrap_frame(self, frame: np.ndarray) -> np.ndarray:
        """Unwrap a single frame using the one-pass OpenCV remap."""
        
        if self.combined_map_x is None:
            h, w = frame.shape[:2]
            self._build_maps(w, h)
        unwrapped = remap_frame(
            frame,
            self.combined_map_x,
            self.combined_map_y,
            self.background,
            self.interpolation,
        )
        return unwrapped
    
    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter out points that are too close to ROI boundary or image edges.
        
        Returns filtered points array.
        """
        if points is None or len(points) == 0:
            return points
        
        filtered_indices = []
        for i, point in enumerate(points):
            x, y = point.ravel()
            
            # Check distance from image edges
            if (x < self.edge_margin or 
                x >= self.top_size - self.edge_margin or
                y < self.edge_margin or 
                y >= self.top_size - self.edge_margin):
                continue
            
            # Check if point is in safe ROI zone (if ROI exists)
            if self.roi_safe is not None:
                xi, yi = int(round(x)), int(round(y))
                xi = max(0, min(xi, self.top_size - 1))
                yi = max(0, min(yi, self.top_size - 1))
                if self.roi_safe[yi, xi] == 0:
                    continue
            
            filtered_indices.append(i)
        
        if len(filtered_indices) == len(points):
            return points
        
        return points[filtered_indices]
    
    def estimate_motion(self, old_points, new_points):
        """Estimate translation and rotation from point correspondences."""
        if len(old_points) < 2:
            return 0.0, 0.0, 0.0
        center_x = self.top_size / 2.0
        center_y = self.top_size / 2.0
        motions = new_points - old_points
        dx = float(np.mean(motions[:, 0]))
        dy = float(np.mean(motions[:, 1]))
        rotations = []
        for old_pt, motion in zip(old_points, motions):
            rx = old_pt[0] - center_x
            ry = old_pt[1] - center_y
            dist = np.sqrt(rx*rx + ry*ry)
            if dist > 10:
                motion_x = motion[0] - dx
                motion_y = motion[1] - dy
                perp_motion = (-ry * motion_x + rx * motion_y) / dist
                theta = perp_motion / dist
                rotations.append(theta)
        rotation = float(np.median(rotations)) if rotations else 0.0
        return dx, dy, rotation
    
    def initialize_tracking(self, unwrapped_frame: np.ndarray) -> None:
        """Initialize feature tracking on the first frame."""
        if len(unwrapped_frame.shape) == 3:
            gray = cv2.cvtColor(unwrapped_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = unwrapped_frame
        self.prev_gray = gray.copy()
        self.p0 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=self.roi_mask,
            **self.feature_params
        )
        # Filter initial points
        if self.p0 is not None:
            self.p0 = self.filter_points(self.p0)
        self.mask = np.zeros_like(unwrapped_frame)
        self.last_update_time = time.time()
        self.total_x = 0.0
        self.total_y = 0.0
        self.total_rotation = 0.0
        if self.p0 is not None:
            print(f"Initial features detected: {len(self.p0)} (after filtering)")
        else:
            print("WARNING: No features detected on first frame!")
    
    def create_roi_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create a debug overlay showing ROI mask and feature points."""
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            overlay = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGRA2BGR)
        else:
            overlay = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.roi_mask is None:
            return overlay
        roi_color = np.zeros_like(overlay)
        roi_color[self.roi_mask > 0] = [0, 0, 255]
        cv2.addWeighted(roi_color, 0.3, overlay, 0.7, 0, overlay)
        roi_contours, _ = cv2.findContours(
            self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, roi_contours, -1, (0, 255, 255), 2)
        if self.p0 is not None:
            for point in self.p0:
                x, y = point.ravel().astype(int)
                x = max(0, min(x, overlay.shape[1] - 1))
                y = max(0, min(y, overlay.shape[0] - 1))
                cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(overlay, (x, y), 7, (0, 255, 0), 1)
        return overlay
    
    def track_displacement(self, unwrapped_frame: np.ndarray) -> tuple:
        """Track displacement and rotation relative to starting position."""
        if len(unwrapped_frame.shape) == 3:
            gray = cv2.cvtColor(unwrapped_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = unwrapped_frame
        
        if self.p0 is None or self.prev_gray is None:
            self.initialize_tracking(unwrapped_frame)
            if self.debug_mode:
                return 0.0, 0.0, 0.0, self.create_roi_debug_overlay(unwrapped_frame.copy())
            return 0.0, 0.0, 0.0, unwrapped_frame.copy()
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.p0, None, **self.lk_params
        )
        
        if p1 is None or len(p1[st == 1]) < 2:
            self.initialize_tracking(unwrapped_frame)
            if self.debug_mode:
                return self.total_x, self.total_y, np.degrees(self.total_rotation), self.create_roi_debug_overlay(unwrapped_frame.copy())
            return self.total_x, self.total_y, np.degrees(self.total_rotation), unwrapped_frame.copy()
        
        good_new = p1[st == 1].reshape(-1, 2)
        good_old = self.p0[st == 1].reshape(-1, 2)
        
        # Filter points that are too close to boundaries
        good_new = self.filter_points(good_new)
        good_old = self.p0[st == 1].reshape(-1, 2)
        # Re-filter good_old to match good_new indices
        st_flat = st.flatten()
        valid_indices = np.where(st_flat == 1)[0]
        good_old = self.p0[valid_indices].reshape(-1, 2)
        # Apply same filter to good_old
        good_old = self.filter_points(good_old)
        # Ensure both arrays have same length
        min_len = min(len(good_new), len(good_old))
        good_new = good_new[:min_len]
        good_old = good_old[:min_len]
        
        if len(good_new) < 2:
            self.initialize_tracking(unwrapped_frame)
            if self.debug_mode:
                return self.total_x, self.total_y, np.degrees(self.total_rotation), self.create_roi_debug_overlay(unwrapped_frame.copy())
            return self.total_x, self.total_y, np.degrees(self.total_rotation), unwrapped_frame.copy()
        
        dx, dy, dtheta = self.estimate_motion(good_old, good_new)
        self.total_x += dx
        self.total_y += dy
        self.total_rotation += dtheta
        
        frame_vis = unwrapped_frame.copy()
        if self.debug_mode and self.roi_mask is not None:
            frame_vis = self.create_roi_debug_overlay(frame_vis)
        
        for new, old in zip(good_new, good_old):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            self.mask = cv2.line(self.mask, (a, b), (c, d), (0, 255, 0), 2)
            frame_vis = cv2.circle(frame_vis, (a, b), 5, (0, 0, 255), -1)
        
        rotation_deg = np.degrees(self.total_rotation)
        cv2.putText(frame_vis, f"Delta: ({dx:.2f}, {dy:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame_vis, f"Position: X={self.total_x:.1f} Y={self.total_y:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(frame_vis, f"Rotation: {rotation_deg:.1f} deg",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_vis, f"Features: {len(good_new)}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if self.roi_mask is not None:
            active_pct = 100 * np.count_nonzero(self.roi_mask) / self.roi_mask.size
            cv2.putText(frame_vis, f"ROI: {active_pct:.0f}%",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.prev_gray = gray.copy()
        # Keep only successfully tracked and filtered points
        self.p0 = p1[st == 1].reshape(-1, 1, 2)
        self.p0 = self.filter_points(self.p0)
        
        return self.total_x, self.total_y, rotation_deg, frame_vis
    
    def maybe_update_features(self, current_time: float) -> bool:
        """Update features if count drops below min_features or every 5 seconds."""
        if self.last_update_time is None:
            return False
        
        # Check if we need to update due to low feature count
        current_count = len(self.p0) if self.p0 is not None else 0
        needs_update = current_count < self.min_features
        
        # Also update every 5 seconds
        if current_time - self.last_update_time >= 5:
            needs_update = True
        
        if needs_update:
            if self.prev_gray is not None:
                self.p0 = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    mask=self.roi_mask,
                    **self.feature_params
                )
                # Filter points
                if self.p0 is not None:
                    self.p0 = self.filter_points(self.p0)
                if self.mask is not None:
                    self.mask[:] = 0
                self.last_update_time = current_time
                if self.p0 is not None:
                    print(f"[{self.frame_count}] Features updated: {len(self.p0)} points (was {current_count})")
                else:
                    print(f"[{self.frame_count}] WARNING: No features detected on update!")
                return True
        return False
    
    def reset(self) -> None:
        """Reset tracking state."""
        self.prev_gray = None
        self.p0 = None
        self.last_update_time = None
        self.frame_count = 0
        self.total_x = 0.0
        self.total_y = 0.0
        self.total_rotation = 0.0
        if self.mask is not None:
            self.mask[:] = 0


def main():
    """Main localization loop using camera or video input."""
    locator = ConicalLocalization(
        rotation_deg=0.0,
        top_size=400,
        field_scale=0.70,
        roi="Images/1.png",
        debug_mode=True,
        background=(0, 0, 0),
        
        roi_margin=10,
        edge_margin=30,
        min_features=7,
    )
    
    # Initialize camera or video source
    if USE_PICAMERA:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("Camera initialized")
    else:
        cap = cv2.VideoCapture("Images/vid2.mp4")
        if not cap.isOpened():
            print("Error: Cannot open video file")
            return
        print("Video file loaded")
    
    print("\nVisualization legend:")
    print("  RED overlay = active ROI area")
    print("  GREEN circles = valid feature points (inside ROI)")
    print("  YELLOW contour = ROI boundary")
    print("  Press ESC to exit\n")
    
    try:
        while True:
            # Capture frame
            if USE_PICAMERA:
                frame = picam2.capture_array()
                # picamera2 returns RGB, convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
            
            # Process frame
            unwrapped = locator.unwrap_frame(frame)
            total_x, total_y, rotation_deg, frame_with_tracks = locator.track_displacement(unwrapped)
            
            current_time = time.time()
            locator.maybe_update_features(current_time)
            locator.frame_count += 1
            
            # Display results
            cv2.imshow('Unwrapped View (with ROI debug)', frame_with_tracks)
            cv2.imshow('Original Frame', frame)
            
            if cv2.waitKey(30) == 27:
                break
                
    finally:
        if USE_PICAMERA:
            picam2.stop()
            picam2.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {locator.frame_count}")
        print(f"Final position: X={locator.total_x:.2f}, Y={locator.total_y:.2f}")
        print(f"Final rotation: {np.degrees(locator.total_rotation):.2f} degrees")

if __name__ == "__main__":
    main()