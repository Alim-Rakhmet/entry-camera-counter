# Entry Camera People Counter & PPE Classifier

## Approach

This pipeline processes construction site video to count personnel entering and exiting a checkpoint, while optionally classifying their Personal Protective Equipment (PPE) using a Vision-Language Model (VLM).

### Detector and Tracker
I chose **YOLOv8 Nano (yolov8n.pt)** integrated with **ByteTrack**. 
* **Why YOLOv8:** It is extremely fast, highly accurate for human detection (`class=0`), and provides built-in tracking.
* **Why ByteTrack:** Unlike simple IoU trackers, ByteTrack handles occlusion well by associating high and low-confidence detection boxes, reducing the chance of ID-switching (which would cause double-counting).

### Direction Logic & Heuristics
Tracks are updated frame-by-frame. Once a track disappears for more than a set threshold (e.g., 30 frames), it is considered "completed" and analyzed:
1. **Displacement Vector:** I calculate the net vector from the track's first recorded coordinate to its last.
2. **Thresholds:** A track must traverse a minimum distance (e.g., `150 pixels`) and exist for a minimum duration (e.g., `1.5 seconds`) to filter out static security guards, camera noise, or short false-positives.
3. **Classification:** * `entry`: Movement predominantly from the upper-left toward the lower-right ($\Delta x > 0$ and $\Delta y > 0$).
   * `exit`: Movement predominantly from the lower-right to the upper-left ($\Delta x < 0$ and $\Delta y < 0$).
   * `unknown`: Static tracks, short tracks, or movement that is purely horizontal/vertical, which defies the expected entry/exit flow.

### Frame Sampling
The script processes every **2nd frame** (`FRAME_SKIP = 2`). 
* **Tradeoff:** This effectively halves processing time and treats the video as 10 FPS instead of 20 FPS. While this makes tracking slightly harder during rapid movements, walking speed through a checkpoint is slow enough that 10 FPS is highly reliable.

### VLM Bonus Integration
For the PPE classification, the pipeline saves the largest bounding box crop (highest resolution) of each tracked person. Once the track finishes, this crop is sent to **Gemini 1.5 Flash** (via `google-generativeai`). I use a strictly typed JSON prompt to ensure the output can be parsed safely into the CSV.

### Known Limitations & Failure Cases
* **Occlusion at the choke point:** If multiple workers walk in tightly packed, YOLO might merge their bounding boxes or swap IDs, leading to undercounting or misattributed entry/exits.
* **Edge-of-frame tracking:** People loitering at the very edge of the camera might generate multiple short tracks if they step in and out of the frame.
* **VLM motion blur:** If the largest bounding box happens to be blurry, the VLM defaults to `unclear`.

### Future Improvements (With More Time)
1. **Perspective Transformation:** Use a homography matrix to map camera pixels to a top-down 2D bird's-eye view. This would make velocity and distance thresholds uniform regardless of distance from the camera.
2. **Line-Crossing Logic:** Instead of vector displacement, draw a precise polygonal zone or a physical crossing line in the frame.
3. **Batch VLM Processing:** Instead of awaiting the VLM API sequentially per track, queue the images and process them asynchronously to speed up total execution.