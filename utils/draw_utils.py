# utils/draw_utils.py

import cv2
import numpy as np

# --- color cont ---
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def draw_tracks(frame, tracks, suspect_id=None):
    frame_h, frame_w, _ = frame.shape
    suspect_bbox = None
    for tid, bbox in tracks:
        x1, y1, x2, y2 = map(int, bbox)
	#suspect
        if tid == suspect_id:
            suspect_bbox = (x1, y1, x2, y2)
            color = RED
            thickness = 3
            label = f"SUSPECT ID: {tid}"
            font_scale = 0.7
	#regular person
        else:
            color = GREEN
            thickness = 2
            label = f"ID: {tid}"
            font_scale = 0.6
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - label_height -baseline), (x1 + label_width, y1), color, -1)
#	cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thickness)
#    return frame
        # --- NEW: Logic for Picture-in-Picture (PiP) Suspect View ---
    if suspect_bbox is not None:
        x1, y1, x2, y2 = suspect_bbox
        
        # 1. Define the zoom area around the suspect
        w = x2 - x1
        h = y2 - y1
        center_x, center_y = x1 + w // 2, y1 + h // 2
        
        # Create a crop area that is 3x the size of the suspect's bbox
        crop_w, crop_h = w * 2, h * 2
        
        # 2. Calculate the crop coordinates
        crop_x1 = max(0, center_x - crop_w // 2)
        crop_y1 = max(0, center_y - crop_h // 2)
        crop_x2 = min(frame_w, center_x + crop_w // 2)
        crop_y2 = min(frame_h, center_y + crop_h // 2)

        # 3. Ensure the crop coordinates are within the frame boundaries
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(frame_w, crop_x2)
        crop_y2 = min(frame_h, crop_y2)

        # 4. Extract the cropped region from the frame
        suspect_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # 5. Resize the crop to a fixed size for the PiP window
        pip_w, pip_h = 250, 250 # Size of the PiP window
        if suspect_crop.shape[0] > 0 and suspect_crop.shape[1] > 0:
            resized_crop = cv2.resize(suspect_crop, (pip_w, pip_h))
            
            # 6. Define position for the PiP window (top-right corner)
            pip_x, pip_y = frame_w - pip_w - 10, 10
            
            # 7. Draw a border and label for the PiP window
            cv2.rectangle(frame, (pip_x - 2, pip_y - 22), (pip_x + pip_w + 2, pip_y + pip_h + 2), BLACK, -1)
            cv2.rectangle(frame, (pip_x - 2, pip_y - 22), (pip_x + pip_w + 2, pip_y + pip_h + 2), WHITE, 2)
            cv2.putText(frame, "Suspect View", (pip_x, pip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            # 8. "Paste" the resized crop onto the main frame
            frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = resized_crop

    return frame

"""
def draw_metrics(frame, mota, idf1, fps):
    text = f"MOTA: {mota:.3f}  IDF1: {idf1:.3f}  FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return frame
"""
def draw_metrics(frame, fps):
    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return frame
