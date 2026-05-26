import os
import cv2
import csv
import json
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import google.generativeai as genai
from typing import Dict, Any

load_dotenv()

VIDEO_PATH = "data/entry_camera.mp4"
OUTPUT_DIR = "results"
FPS = 20
FRAME_SKIP = 2 
EFFECTIVE_FPS = FPS / FRAME_SKIP

MIN_TRACK_LIFETIME_SEC = 1.0 
MIN_DISPLACEMENT_PX = 150     
MAX_DISAPPEAR_FRAMES = 30     

ENABLE_VLM_BONUS = True
VLM_MODEL_NAME = "gemini-1.5-flash"

def classify_ppe_with_vlm(crop_img: np.ndarray) -> Dict[str, Any]:
    if crop_img is None or crop_img.size == 0:
        return {"helmet": "unclear", "safety_vest": "unclear", "vlm_confidence": 0.0, "notes": "Empty crop"}

    success, encoded_image = cv2.imencode('.jpg', crop_img)
    if not success:
        return {"helmet": "unclear", "safety_vest": "unclear", "vlm_confidence": 0.0, "notes": "Encoding failed"}

    try:
        model = genai.GenerativeModel(VLM_MODEL_NAME)
        prompt = (
            "Analyze this crop of a person on a construction site. "
            "Determine if they are wearing a hard hat (helmet) and a safety vest. "
            "Return strictly a JSON object with these exact keys: "
            "'helmet' (yes/no/unclear), 'safety_vest' (yes/no/unclear), "
            "'vlm_confidence' (float 0.0 to 1.0), and 'notes' (brief reason)."
        )
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": encoded_image.tobytes()}
        ])
        
        text = response.text.replace('```json', '').replace('```', '').strip()
        result = json.loads(text)
        return result
    except Exception as e:
        return {"helmet": "unclear", "safety_vest": "unclear", "vlm_confidence": 0.0, "notes": f"API Error: {str(e)}"}

def main():
    global ENABLE_VLM_BONUS
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if ENABLE_VLM_BONUS:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not found. Disabling VLM bonus.")
            ENABLE_VLM_BONUS = False
        else:
            genai.configure(api_key=api_key)

    print("Loading YOLOv8 model")
    model = YOLO('yolov8n.pt') 

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {VIDEO_PATH}")

    active_tracks = defaultdict(lambda: {"positions": [], "frames": [], "best_crop": None, "max_area": 0, "last_seen": 0})
    completed_events = []
    ppe_events = []

    frame_count = 0
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processing {frame_count} / 36000 frames")
        
        if frame_count % FRAME_SKIP != 0:
            continue

        current_time_s = frame_count / FPS

        results = model.track(frame, classes=[0], conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_active_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                
                current_active_ids.add(track_id)
                track = active_tracks[track_id]
                
                track["positions"].append((cx, cy))
                track["frames"].append(current_time_s)
                track["last_seen"] = frame_count
                track["confidence"] = conf

                if area > track["max_area"] and ENABLE_VLM_BONUS:
                    track["max_area"] = area
                    h, w = frame.shape[:2]
                    px1, py1 = max(0, x1-10), max(0, y1-10)
                    px2, py2 = min(w, x2+10), min(h, y2+10)
                    track["best_crop"] = frame[py1:py2, px1:px2].copy()

        stale_ids = []
        for tid, track in active_tracks.items():
            if frame_count - track["last_seen"] > MAX_DISAPPEAR_FRAMES:
                stale_ids.append(tid)

        for tid in stale_ids:
            process_completed_track(tid, active_tracks[tid], completed_events, ppe_events)
            del active_tracks[tid]

    for tid, track in active_tracks.items():
        process_completed_track(tid, track, completed_events, ppe_events)

    cap.release()
    save_results(completed_events, ppe_events, frame_count)

def process_completed_track(track_id, track, completed_events, ppe_events):
    start_time = track["frames"][0]
    end_time = track["frames"][-1]
    duration = end_time - start_time
    
    if duration < MIN_TRACK_LIFETIME_SEC:
        direction = "unknown"
        notes = "Track too short"
    else:
        start_pt = track["positions"][0]
        end_pt = track["positions"][-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < MIN_DISPLACEMENT_PX:
            direction = "unknown"
            notes = "Insufficient displacement"
        else:
            if dx > 0 and dy > 0:
                direction = "entry"
                notes = "Valid entry"
            elif dx < 0 and dy < 0:
                direction = "exit"
                notes = "Valid exit"
            else:
                direction = "unknown"
                notes = "Ambiguous trajectory"

    completed_events.append({
        "track_id": track_id,
        "direction": direction,
        "start_time_s": round(start_time, 2),
        "end_time_s": round(end_time, 2),
        "confidence": round(track.get("confidence", 0.0), 3),
        "notes": notes
    })

    if ENABLE_VLM_BONUS and direction in ["entry", "exit"]:
        vlm_res = classify_ppe_with_vlm(track["best_crop"])
        ppe_events.append({
            "track_id": track_id,
            "direction": direction,
            "timestamp_s": round((start_time + end_time) / 2, 2),
            "helmet": vlm_res.get("helmet", "unclear"),
            "safety_vest": vlm_res.get("safety_vest", "unclear"),
            "vlm_confidence": vlm_res.get("vlm_confidence", 0.0),
            "vlm_notes": vlm_res.get("notes", "")
        })

def save_results(completed_events, ppe_events, total_frames):
    print("\nSaving results...")
    
    entries = sum(1 for e in completed_events if e["direction"] == "entry")
    exits = sum(1 for e in completed_events if e["direction"] == "exit")
    unknowns = sum(1 for e in completed_events if e["direction"] == "unknown")

    summary = {
        "video": VIDEO_PATH.split("/")[-1],
        "entries": entries,
        "exits": exits,
        "unknown_tracks": unknowns,
        "frame_sample_rate": FRAME_SKIP,
        "notes": "Tracks < 1.5s or moving < 150px are discarded as unknown. VLM only triggers on valid entry/exits."
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "events.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "direction", "start_time_s", "end_time_s", "confidence", "notes"])
        writer.writeheader()
        writer.writerows(completed_events)

    if ENABLE_VLM_BONUS and ppe_events:
        with open(os.path.join(OUTPUT_DIR, "ppe_events.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "direction", "timestamp_s", "helmet", "safety_vest", "vlm_confidence", "vlm_notes"])
            writer.writeheader()
            writer.writerows(ppe_events)

    print(f"Done! Summary: {entries} Entries, {exits} Exits, {unknowns} Unknown.")

if __name__ == "__main__":
    main()