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

8. **All angle metrics are computed in pixel space, not normalized coordinates.**
   MediaPipe returns normalized x/y in [0, 1] relative to frame width and height independently. Computing angles directly from normalized coordinates introduces aspect-ratio distortion on non-square frames: on portrait video (1080x1920) a true 45-degree angle measures as approximately 29 degrees in normalized space. All metrics in Phase 4 convert to pixel coordinates before angle calculation.

9. **Lateral trunk tilt is unreliable from a pure side-angle camera.**
   When the pitcher is filmed side-on, the two shoulders appear nearly stacked (one in front of the other), yielding a small apparent horizontal separation — sometimes as little as 2-3% of frame width. Small absolute errors in landmark detection produce large errors in the computed angle. Lateral tilt measurements from side-angle video should be treated as qualitative indicators, not precise values.

10. **Trunk tilt forward sign depends on pitcher facing direction.**
    The metric reports the angle of the shoulder midpoint to hip midpoint line relative to vertical. The sign of "forward" (toward the plate) depends on whether the pitcher faces left or right in the frame. Positive = hip midpoint is to the right of the shoulder midpoint. Users must verify the sign against the video to determine whether positive or negative corresponds to forward lean for their footage.
