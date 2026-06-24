# #!/usr/bin/env python3
# """
# Cross detection on conical-mirror unwrapped video with fixed polarity.
# Parameters: binarize_threshold=140, scale=1.0, match_threshold=0.79
# """
# import cv2
# import numpy as np
# import math
# import time
# from pathlib import Path
# import sys
# from typing import Union, List

# sys.path.append(str(Path(__file__).parent))
# from plane import (
#     build_combined_maps,
#     remap_frame,
#     DEFAULTS
# )


# def load_roi_mask(roi_source, target_size: int) -> np.ndarray | None:
#     if roi_source is None:
#         return None
#     if isinstance(roi_source, (str, Path)):
#         mask = cv2.imread(str(roi_source), cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             return None
#     elif isinstance(roi_source, np.ndarray):
#         if len(roi_source.shape) == 3:
#             mask = cv2.cvtColor(roi_source, cv2.COLOR_BGR2GRAY)
#         else:
#             mask = roi_source.copy()
#     else:
#         return None
#     if mask.shape[0] != target_size or mask.shape[1] != target_size:
#         mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#     _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
#     return mask


# def make_synthetic_cross(size: int, thickness: int | None = None) -> np.ndarray:
#     template = np.zeros((size, size), dtype=np.uint8)
#     center = size // 2
#     if thickness is None:
#         thickness = max(2, size // 6)
#     cv2.line(template, (0, center), (size - 1, center), 255, thickness)
#     cv2.line(template, (center, 0), (center, size - 1), 255, thickness)
#     return template


# def detect_polarity_by_mean(gray: np.ndarray, roi_mask: np.ndarray | None = None) -> str:
#     if roi_mask is not None:
#         if roi_mask.shape != gray.shape:
#             roi_mask = cv2.resize(roi_mask, (gray.shape[1], gray.shape[0]),
#                                   interpolation=cv2.INTER_NEAREST)
#         pixels = gray[roi_mask > 0]
#     else:
#         pixels = gray.ravel()
    
#     if len(pixels) == 0:
#         return 'dark_on_light'
    
#     mean_brightness = float(np.mean(pixels))
    
#     if mean_brightness >= 127:
#         return 'dark_on_light'
#     else:
#         return 'light_on_dark'


# def binarize_and_clean(
#     gray: np.ndarray,
#     threshold: int = 140,
#     morph_kernel_size: int = 2,
#     morph_iterations: int = 1,
#     polarity: str = 'dark_on_light',
# ) -> np.ndarray:
#     if gray.dtype != np.uint8:
#         gray = np.clip(gray, 0, 255).astype(np.uint8)
    
#     if polarity == 'dark_on_light':
#         thresh_type = cv2.THRESH_BINARY_INV
#     else:
#         thresh_type = cv2.THRESH_BINARY
    
#     _, binary = cv2.threshold(gray, threshold, 255, thresh_type)
    
#     if morph_kernel_size > 0 and morph_iterations > 0:
#         kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
#         binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
#         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
#     return binary


# def verify_cross_at_point(binary: np.ndarray, x: int, y: int, radius: int = 8) -> float:
#     h, w = binary.shape[:2]
#     directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
#     line_scores = []
    
#     for dx, dy in directions:
#         white_count = 0
#         total = 0
#         for i in range(1, radius + 1):
#             px = x + dx * i
#             py = y + dy * i
#             if 0 <= px < w and 0 <= py < h:
#                 if binary[py, px] > 127:
#                     white_count += 1
#                 total += 1
#         if total > 0:
#             line_scores.append(white_count / total)
#         else:
#             line_scores.append(0.0)
    
#     return min(line_scores) if line_scores else 0.0


# def rotate_template(template: np.ndarray, angle_deg: float) -> np.ndarray:
#     h, w = template.shape[:2]
#     center = (w / 2.0, h / 2.0)
#     M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
#     cos = abs(M[0, 0])
#     sin = abs(M[0, 1])
#     new_w = int(h * sin + w * cos)
#     new_h = int(h * cos + w * sin)
#     M[0, 2] += (new_w - w) / 2.0
#     M[1, 2] += (new_h - h) / 2.0
#     rotated = cv2.warpAffine(
#         template, M, (new_w, new_h),
#         borderMode=cv2.BORDER_CONSTANT, borderValue=0
#     )
#     return rotated


# def scale_template(template: np.ndarray, scale: float) -> np.ndarray:
#     if abs(scale - 1.0) < 1e-6:
#         return template
#     h, w = template.shape[:2]
#     new_w = max(3, int(w * scale))
#     new_h = max(3, int(h * scale))
#     return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)


# def build_template_bank(
#     base_templates: list[np.ndarray],
#     angles: list[float],
#     scales: list[float],
# ) -> list[np.ndarray]:
#     bank = []
#     for base in base_templates:
#         if base.max() > 1 and base.dtype != np.uint8:
#             _, base_bin = cv2.threshold(base, 127, 255, cv2.THRESH_BINARY)
#         else:
#             base_bin = base
#         for scale in scales:
#             scaled = scale_template(base_bin, scale)
#             for angle in angles:
#                 rotated = rotate_template(scaled, angle)
#                 if rotated.shape[0] < 3 or rotated.shape[1] < 3:
#                     continue
#                 bank.append(rotated)
#     return bank


# class CrossDetector:
#     def __init__(
#         self,
#         cx: float = DEFAULTS["cx"],
#         cy: float = DEFAULTS["cy"],
#         outer_r: float = DEFAULTS["outer_r"],
#         lens_deg: float = DEFAULTS["lens_deg"],
#         cone_power: float = DEFAULTS["cone_power"],
#         rotation_deg: float = DEFAULTS["rotation_deg"],
#         top_size: int = DEFAULTS["top_size"],
#         field_scale: float = DEFAULTS["field_scale"],
#         background: tuple[int, int, int] = (0, 0, 0),
#         interpolation: int = cv2.INTER_LINEAR,
#         roi: str | Path | np.ndarray | None = None,
#         templates: Union[str, Path, List[Union[str, Path]], np.ndarray, List[np.ndarray], None] = None,
#         template_size: int = 24,
#         angles: list[float] | None = None,
#         scales: list[float] | None = None,
#         binarize_threshold: int = 140,
#         morph_kernel_size: int = 2,
#         morph_iterations: int = 1,
#         match_threshold: float = 0.79,
#         nms_radius: int = 15,
#         cross_verification_radius: int = 8,
#         cross_verification_threshold: float = 0.6,
#         scale_factor: float = 1.0,
#         marker_color: tuple[int, int, int] = (0, 0, 255),
#         marker_radius: int = 5,
#         show_roi: bool = False,
#         debug_binarization: bool = True,
#         fixed_polarity: str | None = None,
#     ):
#         self.cx = cx
#         self.cy = cy
#         self.outer_r = outer_r
#         self.lens_deg = lens_deg
#         self.cone_power = cone_power
#         self.rotation_deg = rotation_deg
#         self.top_size = top_size
#         self.field_scale = field_scale
#         self.background = background
#         self.interpolation = interpolation
#         self.scale_factor = scale_factor
#         self.show_roi = show_roi
#         self.debug_binarization = debug_binarization
#         self.marker_color = marker_color
#         self.marker_radius = marker_radius
#         self.match_threshold = match_threshold
#         self.nms_radius = nms_radius
#         self.binarize_threshold = binarize_threshold
#         self.morph_kernel_size = morph_kernel_size
#         self.morph_iterations = morph_iterations
#         self.cross_verification_radius = cross_verification_radius
#         self.cross_verification_threshold = cross_verification_threshold
#         self.fixed_polarity = fixed_polarity
#         self._detected_polarity = None

#         self.roi_mask = load_roi_mask(roi, top_size)

#         base_templates = self._load_templates(templates, template_size)

#         if angles is None:
#             angles = [0, 15, 30, 45, 60, 75]
#         if scales is None:
#             scales = [1.0]

#         self.template_bank = build_template_bank(base_templates, angles, scales)
#         print(f"Binary template bank: {len(self.template_bank)} templates "
#               f"({len(base_templates)} base × {len(angles)} angles × {len(scales)} scales)")
#         print(f"Fixed threshold: {binarize_threshold}")
#         print(f"Match threshold: {match_threshold}, NMS radius: {nms_radius}px")

#         self.combined_map_x = None
#         self.combined_map_y = None

#     def _load_templates(self, templates, default_size: int) -> list[np.ndarray]:
#         if templates is None:
#             return [make_synthetic_cross(default_size)]
#         if isinstance(templates, (str, Path)):
#             templates = [templates]
#         elif isinstance(templates, np.ndarray):
#             templates = [templates]
#         result = []
#         for t in templates:
#             if isinstance(t, (str, Path)):
#                 img = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE)
#                 if img is None:
#                     print(f"Warning: could not load template {t}, skipping")
#                     continue
#                 _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#                 result.append(img_bin)
#             elif isinstance(t, np.ndarray):
#                 if len(t.shape) == 3:
#                     t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
#                 _, t_bin = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)
#                 result.append(t_bin)
#         if not result:
#             print("No valid templates, falling back to synthetic")
#             return [make_synthetic_cross(default_size)]
#         return result

#     def _build_maps(self, frame_width: int, frame_height: int) -> None:
#         self.combined_map_x, self.combined_map_y = build_combined_maps(
#             top_size=self.top_size,
#             source_width=frame_width,
#             source_height=frame_height,
#             cx=self.cx, cy=self.cy, outer_r=self.outer_r,
#             rotation_deg=self.rotation_deg, field_scale=self.field_scale,
#             lens_deg=self.lens_deg, cone_power=self.cone_power,
#         )

#     def unwrap(self, frame: np.ndarray) -> np.ndarray:
#         if self.combined_map_x is None:
#             h, w = frame.shape[:2]
#             self._build_maps(w, h)
#         return remap_frame(
#             frame, self.combined_map_x, self.combined_map_y,
#             self.background, self.interpolation,
#         )

#     def find_crosses(self, unwrapped: np.ndarray) -> list:
#         if len(unwrapped.shape) == 3:
#             gray = cv2.cvtColor(unwrapped[:, :, :3], cv2.COLOR_BGR2GRAY)
#         else:
#             gray = unwrapped.copy()

#         if self.scale_factor != 1.0:
#             h, w = gray.shape[:2]
#             new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
#             gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         else:
#             new_h, new_w = gray.shape[:2]

#         roi_scaled = None
#         if self.roi_mask is not None:
#             if self.scale_factor != 1.0:
#                 roi_scaled = cv2.resize(self.roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#             else:
#                 roi_scaled = self.roi_mask
#             gray = cv2.bitwise_and(gray, roi_scaled)

#         # FIXED POLARITY: detect once, then reuse
#         if self._detected_polarity is None:
#             if self.fixed_polarity is not None:
#                 self._detected_polarity = self.fixed_polarity
#                 print(f"Using FIXED polarity: {self._detected_polarity}")
#             else:
#                 self._detected_polarity = detect_polarity_by_mean(gray, roi_scaled)
#                 print(f"Auto-detected polarity (1st frame): {self._detected_polarity}")

#         mean_b = float(np.mean(gray[roi_scaled > 0] if roi_scaled is not None else gray))

#         binary = binarize_and_clean(
#             gray,
#             self.binarize_threshold,
#             self.morph_kernel_size,
#             self.morph_iterations,
#             self._detected_polarity,
#         )
        
#         if self.debug_binarization:
#             cv2.imshow("Gray", gray)
#             cv2.imshow("Binary", binary)

#         candidates = []
#         for template in self.template_bank:
#             th, tw = template.shape[:2]
#             if th > binary.shape[0] or tw > binary.shape[1]:
#                 continue
#             result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
#             matches = np.where(result >= self.match_threshold)
#             for y, x in zip(matches[0], matches[1]):
#                 score = float(result[y, x])
#                 cx = (x + tw / 2.0) / self.scale_factor
#                 cy = (y + th / 2.0) / self.scale_factor
#                 candidates.append((cx, cy, score))

#         candidates.sort(key=lambda c: c[2], reverse=True)

#         final = []
#         for cx, cy, score in candidates:
#             too_close = False
#             for fx, fy in final:
#                 if math.hypot(cx - fx, cy - fy) < self.nms_radius:
#                     too_close = True
#                     break
#             if too_close:
#                 continue
            
#             ix, iy = int(cx), int(cy)
#             if 0 <= ix < binary.shape[1] and 0 <= iy < binary.shape[0]:
#                 cross_score = verify_cross_at_point(
#                     binary, ix, iy, self.cross_verification_radius
#                 )
#                 if cross_score >= self.cross_verification_threshold:
#                     final.append((cx, cy))

#         # Debug output every frame
#         if self.debug_binarization:
#             print(f"Frame: mean={mean_b:.0f}, polarity={self._detected_polarity}, "
#                   f"candidates={len(candidates)}, final={len(final)}")

#         return final

#     def draw(self, frame: np.ndarray, crosses: list) -> np.ndarray:
#         if len(frame.shape) == 3 and frame.shape[2] == 4:
#             out = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGRA2BGR)
#         else:
#             out = frame.copy()

#         r = self.marker_radius
#         for x, y in crosses:
#             if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
#                 cv2.circle(out, (int(x), int(y)), r, self.marker_color, -1)
#                 cv2.circle(out, (int(x), int(y)), r + 2, (255, 255, 255), 1)

#         if self.roi_mask is not None and self.show_roi:
#             roi_color = np.zeros_like(out)
#             roi_color[self.roi_mask > 0] = [0, 0, 255]
#             cv2.addWeighted(roi_color, 0.3, out, 0.7, 0, out)

#         return out

#     def process_frame(self, frame: np.ndarray) -> tuple:
#         unwrapped = self.unwrap(frame)
#         crosses = self.find_crosses(unwrapped)
#         overlay = self.draw(unwrapped, crosses)
#         return unwrapped, crosses, overlay


# def main():
#     detector = CrossDetector(
#         cx=308,
#         cy=234,
#         outer_r=230,
#         lens_deg=-81.86,
#         cone_power=2.245,
#         rotation_deg=-2.0,
#         top_size=400,
#         field_scale=0.70,
#         background=(0, 0, 0),
#         interpolation=cv2.INTER_LINEAR,
#         roi="Images/1.png",
#         show_roi=False,
#         templates=["Images/Cross2.png"],
#         template_size=24,
#         angles=[0, 15, 30, 45, 60, 75],
#         scales=[1.0],
#         binarize_threshold=140,
#         match_threshold=0.79,
#         scale_factor=1.0,
#         morph_kernel_size=2,
#         morph_iterations=1,
#         nms_radius=15,
#         cross_verification_radius=8,
#         cross_verification_threshold=0.6,
#         marker_color=(0, 0, 255),
#         marker_radius=5,
#         debug_binarization=True,
#         fixed_polarity='dark_on_light',  # FIX: зафиксировать полярность
#     )
#     cap = cv2.VideoCapture("Images/vid1.mp4")
#     if not cap.isOpened():
#         print("Error: Cannot open video file")
#         return

#     print("Cross Detection on VIDEO started. Press ESC to exit.")
#     print(f"Bank: {len(detector.template_bank)} templates")
#     print(f"Tuned params: threshold={detector.binarize_threshold}, "
#           f"match={detector.match_threshold}, scale={detector.scale_factor}")
#     print(f"Fixed polarity: {detector.fixed_polarity}\n")

#     frame_count = 0
#     start_time = time.time()

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video reached")
#                 break

#             t0 = time.perf_counter()
#             unwrapped, crosses, overlay = detector.process_frame(frame)
#             dt = time.perf_counter() - t0

#             frame_count += 1
#             fps = 1.0 / dt if dt > 0 else 0
#             cv2.putText(overlay, f"Crosses: {len(crosses)}  FPS: {fps:.0f}",
#                         (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#             cv2.imshow('Cross Detection', overlay)
#             if cv2.waitKey(1) == 27:
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         elapsed = time.time() - start_time
#         print(f"\nProcessed {frame_count} frames in {elapsed:.2f}s")
#         print(f"Average FPS: {frame_count / elapsed:.1f}")


# if __name__ == "__main__":
#     main()

import cv2

# 1. Загрузка изображений (в градациях серого)
img1 = cv2.imread('Images/TESTING_TEMPLATE.jpg', 0)   # Объект (шаблон)
img2 = cv2.imread('Images/TESTING_IMAGE.jpg', 0) # Сцена

# 2. Инициализация ORB
# nfeatures - макс. кол-во точек, scaleFactor, nlevels - параметры пирамиды
orb = cv2.ORB_create(nfeatures=1000)

# 3. Поиск ключевых точек и вычисление дескрипторов
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. Сопоставление (матчинг) дескрипторов
# Для ORB используем NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Сортировка по расстоянию (чем меньше, тем лучше совпадение)
matches = sorted(matches, key=lambda x: x.distance)

# 5. Отрисовка результатов
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imshow('ORB Matches', img_matches)
cv2.waitKey(0)