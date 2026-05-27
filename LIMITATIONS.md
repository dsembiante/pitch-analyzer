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

11. **Stride length is measured as horizontal (x-axis) distance only.**
    The true path the lead foot travels through space is a diagonal arc; measuring only the horizontal component underestimates the physical stride length by a small amount (typically a few percent) and ignores vertical leg-lift height. Horizontal-only measurement is the biomechanics convention because it captures the distance the front foot advanced toward the plate, which is what drives momentum. The measurement is also sensitive to foot-ankle landmark position noise at the start-of-motion and foot-strike frames.

12. **Balance point back-foot position is sampled at leg_lift_peak, not start_of_motion.**
    The drift metric computes hip midpoint relative to the back ankle at the leg lift peak frame. When start_of_motion and leg_lift_peak are the same frame (as in IMG_8605, both at frame 172), this is identical to using start_of_motion. When they differ, the back foot may have begun to pivot, making the reading a slight underestimate of true drift relative to the original stance position.

13. **Ankle landmarks can have low MediaPipe visibility at setup frames.**
    The planted back foot is often partially occluded or at the edge of the pose model's confidence region at the start-of-motion frame. Stride length and balance point use a relaxed visibility threshold (0.25 vs the usual 0.5) for ankle landmarks at these static frames. Position data at this confidence level is usable for position but would be unsuitable for velocity or angle calculations.

14. **Hip-shoulder separation is a 2D projection, not a true rotational measurement.**
    True hip-shoulder separation is the rotation about the spine axis between the hip plane and shoulder plane. The 2D metric computed here measures the apparent angular difference between the hip line and shoulder line as projected onto the camera plane. From a pure side-facing camera, this 2D projection underestimates true separation because the rotation axis is nearly perpendicular to the camera view direction, making it mostly invisible. Values from this metric track relative changes reliably but should not be compared against 3D biomechanics literature benchmarks.

15. **Head movement path length includes the full body stride arc, not just head wobble.**
    The head path length and max deviation metrics are computed from leg_lift_peak to ball_release — the full delivery window. During this period the pitcher's entire body moves forward through the stride, so the head naturally travels a large distance. High path length values partly reflect the stride motion, not just independent head instability. Lateral head deviation (perpendicular to the throwing direction) is the true stability signal; separating lateral from forward head movement requires knowing the throwing direction in the image plane.

16. **Front knee extension rate is sensitive to phase detection timing.**
    The rate of knee extension is computed from foot_strike to ball_release. Errors in either phase detection frame can shift the start or end angle significantly (a 1-frame error at 30fps = ~33ms, during which the knee can rotate several degrees). The direction flag ("extending" vs "flexing") is reliable when the rate exceeds ±100 deg/s; rates near the ±50 deg/s stability threshold should be treated with caution.
