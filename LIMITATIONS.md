# Known Limitations

1. **Max layback is a 2D proxy for true 3D shoulder MER.**
   The detector measures the throwing wrist's furthest horizontal layback position in the camera plane. This correlates with — but does not directly measure — true (3D) shoulder maximum external rotation. Because it is a single-plane measurement, the detected frame can occur slightly before foot strike due to camera projection. This is expected behavior, not a detection error. True 3D MER requires multi-camera setup or markered motion capture.

2. **Hip-shoulder separation (Phase 4) is a 2D approximation.**
   Separation angle is computed from the side-facing camera angle. This underestimates true 3D rotation when the pitcher's torso is not perfectly perpendicular to the camera.

3. **Single camera, side-angle view required.**
   The full body must remain in frame for the entire delivery. Detections degrade if the pitcher is partially cropped, viewed from a non-side angle, or the camera moves during the delivery.

4. **Ball release timing is an approximation.**
   Release is detected as peak throwing-wrist horizontal velocity. The actual moment the ball leaves the hand may be 1-2 frames later due to finger drag and wrist snap that trail the velocity peak.

5. **Single pitch per video assumed.**
   The pitching window detector selects the single highest-sustained wrist velocity burst in the video. Videos containing multiple pitches, bullpen sessions, or warm-up throws will produce unreliable results. Clip to a single delivery before processing.

6. **Pitcher must be the dominant moving subject in frame.**
   The pose estimator tracks the most prominent person. Crowded backgrounds, coaches walking through the frame, or batter movement can cause landmark tracking to jump between subjects, corrupting phase detection.

7. **Pose extraction quality depends on video conditions.**
   Best results require 1080p or higher resolution, 30fps or higher frame rate, good lighting with the pitcher in contrast against the background, and minimal motion blur. Low-resolution or poorly lit video will produce noisy landmark coordinates that degrade all downstream phase detections.
