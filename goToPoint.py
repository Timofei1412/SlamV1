#!/usr/bin/env python3
"""
Go-To-Point controller for differential drive robot.
Uses ConicalLocalization with Picamera2.
"""
import cv2
import numpy as np
import math
import time
import sys
from pathlib import Path

try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False
    print("Warning: picamera2 not found, falling back to cv2.VideoCapture")

from localisation import ConicalLocalization

# ==============================================================================
# НАСТРОЙКИ
# ==============================================================================
TARGET_X = 150.0  
TARGET_Y = 100.0  

KP_LINEAR = 0.8
KP_ANGULAR = 2.5
MAX_LINEAR_V = 50.0
MAX_ANGULAR_W = 1.5

DIST_TOLERANCE = 5.0
ANGLE_TOLERANCE = 0.1

WHEEL_BASE = 0.2

INVERT_Y = False
INVERT_THETA = False
INVERT_MOTORS = False

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def diff_drive_kinematics(v, omega, wheel_base):
    v_left = v - omega * (wheel_base / 2.0)
    v_right = v + omega * (wheel_base / 2.0)
    return v_left, v_right

def send_to_motors(v_left, v_right):
    pass

# ==============================================================================
# ГЛАВНЫЙ ЦИКЛ
# ==============================================================================
def main():
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
    
    # Инициализация камеры
    if USE_PICAMERA:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("Picamera2 initialized")
    else:
        cap = cv2.VideoCapture("Images/vid2.mp4")
        if not cap.isOpened():
            print("Error: cannot open video")
            return

    print(f"Target: ({TARGET_X}, {TARGET_Y})")
    print("Press ESC to exit")

    arrived = False

    try:
        while True:
            # Захват кадра
            if USE_PICAMERA:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Локализация
            unwrapped = locator.unwrap_frame(frame)
            total_x, total_y, rotation_deg, frame_vis = locator.track_displacement(unwrapped)
            
            current_x = total_x
            current_y = -total_y if INVERT_Y else total_y
            current_theta = math.radians(-rotation_deg if INVERT_THETA else rotation_deg)

            # Геометрия
            dx = TARGET_X - current_x
            dy = TARGET_Y - current_y
            distance = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            heading_error = normalize_angle(target_angle - current_theta)

            # Управление
            if distance < DIST_TOLERANCE:
                v_cmd, w_cmd = 0.0, 0.0
                if not arrived:
                    print("🎯 TARGET REACHED!")
                    arrived = True
            else:
                w_cmd = KP_ANGULAR * heading_error
                v_cmd = KP_LINEAR * distance * math.cos(heading_error)

            w_cmd = max(-MAX_ANGULAR_W, min(MAX_ANGULAR_W, w_cmd))
            v_cmd = max(0.0, min(MAX_LINEAR_V, v_cmd))

            # Кинематика
            v_left, v_right = diff_drive_kinematics(v_cmd, w_cmd, WHEEL_BASE)
            if INVERT_MOTORS:
                v_left, v_right = v_right, v_left

            send_to_motors(v_left, v_right)

            # Визуализация
            h, w = frame_vis.shape[:2]
            center = (w // 2, h // 2)
            
            robot_heading_line_end = (
                int(center[0] + 50 * math.cos(current_theta)),
                int(center[1] - 50 * math.sin(current_theta))
            )
            cv2.line(frame_vis, center, robot_heading_line_end, (255, 0, 0), 2)

            scale = 2.0 
            target_pt = (int(center[0] + dx * scale), int(center[1] - dy * scale))
            cv2.line(frame_vis, center, target_pt, (0, 255, 0), 2)
            cv2.circle(frame_vis, target_pt, 8, (0, 255, 255), -1)

            cv2.putText(frame_vis, f"Target: ({TARGET_X:.0f}, {TARGET_Y:.0f})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_vis, f"Dist: {distance:.1f} | Err: {math.degrees(heading_error):.1f} deg", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_vis, f"Cmd: v={v_cmd:.1f}, w={w_cmd:.2f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow('Go-To-Point Debug', frame_vis)
            
            if cv2.waitKey(30) == 27:
                break

    finally:
        if USE_PICAMERA:
            picam2.stop()
            picam2.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        send_to_motors(0.0, 0.0)

if __name__ == "__main__":
    main()