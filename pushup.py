import cv2
import numpy as np
import time
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully!")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available, using motion detection mode")

class FinalBalancedPushUpCounter:
    def __init__(self):
        # Core state
        self.counter = 0
        self.stage = None
        self.last_stage = "UP" # Start assuming user is up
        self.rep_start_time = None
        self.last_rep_time = 0

        # --- TUNING PARAMETERS ---
        # Ready state: User must hold plank for longer (~0.8 seconds)
        self.system_ready = False
        self.stable_frames_required = 25 # Hold plank for 25 frames (Increased from 20)
        self.stable_frame_count = 0
        self.min_rep_interval = 0.8 # Minimum time between reps

        # Rep detection: Need 2 consecutive frames in a state for responsiveness
        self.consecutive_frames_required = 2
        self.consecutive_down_frames = 0
        self.consecutive_up_frames = 0

        # Angles: Define UP and DOWN states - Balanced counting, stricter ready
        self.angle_threshold_up_ready = 160 # Stricter angle to get ready (Increased from 155)
        self.angle_threshold_up = 155 # Angle > 155 degrees is UP for counting
        self.angle_threshold_down = 110 # Angle < 110 degrees is DOWN

        # Smoothing: Use mean for responsiveness, moderate buffer size
        self.smoothing_buffer_size = 8
        self.left_elbow_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.right_elbow_buffer = deque(maxlen=self.smoothing_buffer_size) # Keep separate buffer instance
        self.shoulder_buffer = deque(maxlen=self.smoothing_buffer_size)

        # Confidence: MediaPipe detection confidence
        self.min_detection_confidence = 0.65 # Balanced confidence
        self.min_tracking_confidence = 0.65

        # Quality Check (for summary, not displayed in real-time)
        self.quality_depth_threshold = 95 # Angle < 95 for a "good" rep
        self.quality_speed_min = 0.8
        self.quality_speed_max = 4.0
        self.quality_shoulder_alignment_threshold = 30 # Max pixel difference

        # Performance metrics
        self.good_reps = 0
        self.bad_reps = 0
        self.avg_speed = 0
        self.speeds = []

        # UI
        self.full_screen = False

        # Setup based on availability
        if MEDIAPIPE_AVAILABLE:
            self.setup_mediapipe()
        else:
            self.setup_motion_detection()

    def setup_mediapipe(self):
        """Initializes MediaPipe Pose detection."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 0=light, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detection_mode = "mediapipe"
        print("Using MediaPipe Pose Detection - FINAL BALANCED FRONT VIEW PUSH-UPS")

    def setup_motion_detection(self):
        """Initializes Motion Detection fallback."""
        self.background = None
        self.motion_threshold = 5000
        self.last_motion_time = 0
        self.motion_cooldown = 1.5
        self.consecutive_motion_frames = 0
        self.motion_frames_required = 3
        self.detection_mode = "motion"
        print("Using Motion Detection - FINAL BALANCED FRONT VIEW PUSH-UPS")

    def calculate_angle(self, a, b, c):
        """Calculates the angle between three points (e.g., shoulder, elbow, wrist)."""
        a = np.array(a) # First point
        b = np.array(b) # Mid point (vertex)
        c = np.array(c) # End point

        # Calculate angle using atan2, more stable than acos
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        # Ensure angle is between 0 and 180
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_shoulder_alignment(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Checks if shoulders and hips are relatively level."""
        # Check if input points are valid
        if not all(isinstance(p, (list, np.ndarray)) and len(p) == 2 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return False # Invalid input

        shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_y_diff = abs(left_hip[1] - right_hip[1])
        # Return True if the vertical difference is within the threshold
        return shoulder_y_diff < self.quality_shoulder_alignment_threshold and \
               hip_y_diff < self.quality_shoulder_alignment_threshold

    def smooth_value(self, buffer, new_value):
        """Applies simple averaging smoothing to a value."""
        # Ensure buffer is not empty before calculating mean
        buffer.append(new_value)
        if buffer:
            return np.mean(buffer)
        return new_value # Return new value if buffer was empty

    def detect_pushup_quality(self, min_elbow_angle_during_rep, shoulder_alignment_ok, rep_time):
        """Determines if the completed push-up met quality criteria."""
        depth_ok = min_elbow_angle_during_rep < self.quality_depth_threshold
        form_ok = shoulder_alignment_ok
        speed_ok = self.quality_speed_min < rep_time < self.quality_speed_max

        # Print quality feedback for debugging/info
        # print(f"  Quality Check: DepthOK={depth_ok}({min_elbow_angle_during_rep:.1f}<{self.quality_depth_threshold}), FormOK={form_ok}, SpeedOK={speed_ok}({self.quality_speed_min}<{rep_time:.2f}<{self.quality_speed_max})")

        return depth_ok and form_ok and speed_ok

    def process_mediapipe_frame(self, frame):
        """Processes a single frame using MediaPipe for push-up detection."""
        h, w = frame.shape[:2]
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Performance optimization

        # Make detection
        results = self.pose.process(rgb_frame)

        # Reset angle for this frame
        avg_elbow_angle = 0
        shoulder_alignment_ok = False # Assume not aligned until checked

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates safely
                lm = self.mp_pose.PoseLandmark
                left_shoulder = [landmarks[lm.LEFT_SHOULDER.value].x * w, landmarks[lm.LEFT_SHOULDER.value].y * h]
                left_elbow = [landmarks[lm.LEFT_ELBOW.value].x * w, landmarks[lm.LEFT_ELBOW.value].y * h]
                left_wrist = [landmarks[lm.LEFT_WRIST.value].x * w, landmarks[lm.LEFT_WRIST.value].y * h]
                right_shoulder = [landmarks[lm.RIGHT_SHOULDER.value].x * w, landmarks[lm.RIGHT_SHOULDER.value].y * h]
                right_elbow = [landmarks[lm.RIGHT_ELBOW.value].x * w, landmarks[lm.RIGHT_ELBOW.value].y * h]
                right_wrist = [landmarks[lm.RIGHT_WRIST.value].x * w, landmarks[lm.RIGHT_WRIST.value].y * h]
                left_hip = [landmarks[lm.LEFT_HIP.value].x * w, landmarks[lm.LEFT_HIP.value].y * h]
                right_hip = [landmarks[lm.RIGHT_HIP.value].x * w, landmarks[lm.RIGHT_HIP.value].y * h]

                # Calculate angles
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Smooth the average angle
                current_avg_angle = (left_elbow_angle + right_elbow_angle) / 2
                avg_elbow_angle = self.smooth_value(self.left_elbow_buffer, current_avg_angle) # Use one buffer for avg

                # Check shoulder alignment
                shoulder_alignment_ok = self.calculate_shoulder_alignment(left_shoulder, right_shoulder, left_hip, right_hip)

                current_time = time.time()

                # --- READY STATE LOGIC ---
                if not self.system_ready:
                    # Must be holding an UP pose (stricter angle) to stabilize for longer
                    if avg_elbow_angle > self.angle_threshold_up_ready: # Use stricter ready angle
                        self.stable_frame_count = min(self.stable_frame_count + 1, self.stable_frames_required)
                        if self.stable_frame_count >= self.stable_frames_required: # Use increased frames required
                            self.system_ready = True
                            self.stage = "UP" # Initial stage after ready
                            self.last_stage = "UP"
                            print("✅ System Ready - Start Push-ups!")
                    else:
                        # Reset count if not holding UP pose
                        self.stable_frame_count = max(0, self.stable_frame_count - 1)

                # --- COUNTING LOGIC (Only if system is ready) ---
                if self.system_ready:
                    # Determine current stage based on smoothed angle using BALANCED thresholds
                    current_stage_potential = self.stage # Assume current stage until changed
                    if avg_elbow_angle > self.angle_threshold_up: # Use balanced UP threshold
                        self.consecutive_up_frames += 1
                        self.consecutive_down_frames = 0
                        current_stage_potential = "UP"
                    elif avg_elbow_angle < self.angle_threshold_down: # Use balanced DOWN threshold
                        self.consecutive_down_frames += 1
                        self.consecutive_up_frames = 0
                        current_stage_potential = "DOWN"
                    else: # In between
                        self.consecutive_up_frames = 0
                        self.consecutive_down_frames = 0
                        # Keep current_stage_potential as self.stage

                    # Confirm stage change only after consecutive frames
                    if current_stage_potential == "UP" and self.consecutive_up_frames >= self.consecutive_frames_required:
                        # --- Rep completed on transition DOWN -> UP ---
                        if self.stage == "DOWN":
                            self.counter += 1
                            if self.rep_start_time:
                                rep_time = current_time - self.rep_start_time
                                # Basic check for reasonable rep time
                                if 0.3 < rep_time < 10.0: # Filter out absurdly fast/slow reps
                                    self.speeds.append(rep_time)
                                    self.avg_speed = np.mean(self.speeds[-10:]) if self.speeds else 0

                                    # Check quality using min angle from buffers during the rep
                                    # Ensure buffer is not empty before calling min
                                    min_angle_this_rep = min(list(self.left_elbow_buffer)) if self.left_elbow_buffer else avg_elbow_angle
                                    if self.detect_pushup_quality(min_angle_this_rep, shoulder_alignment_ok, rep_time):
                                        self.good_reps += 1
                                        print(f"✅ Rep #{self.counter} (Good) - Time: {rep_time:.2f}s")
                                    else:
                                        self.bad_reps += 1
                                        print(f"⚠️ Rep #{self.counter} (Bad) - Time: {rep_time:.2f}s")
                                else:
                                     print(f"⚠️ Rep #{self.counter} (Ignored - Unrealistic Time: {rep_time:.2f}s)")


                                self.last_rep_time = current_time
                            self.rep_start_time = None # Reset start time after finishing UP state

                        self.stage = "UP"


                    elif current_stage_potential == "DOWN" and self.consecutive_down_frames >= self.consecutive_frames_required:
                         # --- Start of rep on transition UP -> DOWN ---
                        if self.stage == "UP" and (current_time - self.last_rep_time > self.min_rep_interval):
                            self.rep_start_time = current_time # Start timer for rep duration
                            # Clear buffer only when starting down to capture the minimum angle of *this* rep
                            self.left_elbow_buffer.clear()
                            # self.right_elbow_buffer.clear() # Only using one buffer now
                            print(f"🏋️ Rep #{self.counter + 1} - Going down...")
                            self.stage = "DOWN"
                        elif self.stage == "UP":
                            # Still in UP stage, but angle dropped below threshold - maybe starting down but too soon?
                            # Do nothing here, wait for rep interval or angle to go back up
                            pass


                # --- DRAWING ---
                # Draw landmarks with updated colors
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    # Joints: Dark Magenta (like squats/lunges)
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    # Connections: Orange (like squats/lunges)
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
                )

                # Draw angle info near elbows (using raw angle for immediate feedback)
                cv2.putText(frame, f"{int(left_elbow_angle)}",
                            tuple(np.add(np.multiply(left_elbow, [1, 1]).astype(int), [10, -10])), # Offset text
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(right_elbow_angle)}",
                            tuple(np.add(np.multiply(right_elbow, [1, 1]).astype(int), [-40, -10])), # Offset text differently
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                # Display UI info based on smoothed angle
                self.display_pushup_info(frame, avg_elbow_angle, shoulder_alignment_ok)

            else:
                 # --- NO POSE DETECTED ---
                self.system_ready = False
                self.stable_frame_count = 0
                self.stage = None # Reset stage if pose is lost
                # Draw *only* the error text
                cv2.putText(frame, 'NO POSE DETECTED', (int(w*0.1), int(h*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Face camera directly', (int(w*0.1), int(h*0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing frame: {e}")
            # Attempt to draw basic UI even on error, if possible
            try:
                self.display_pushup_info(frame, 0, False)
            except Exception as display_e:
                print(f"Error displaying info after frame error: {display_e}")


        return frame

    def display_pushup_info(self, frame, elbow_angle, shoulder_alignment_ok):
        """Displays the UI elements on the frame."""
        h, w = frame.shape[:2]

        # --- Top Left Info Panel ---
        info_x = int(w * 0.02)
        info_y_start = int(h * 0.08)
        line_height = int(h * 0.05)

        # Counter
        cv2.putText(frame, f'PUSH-UPS: {self.counter}', (info_x, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        current_y = info_y_start + int(line_height * 1.4) # Slightly larger gap

        # Stage
        stage_text = f'STAGE: {self.stage}' if self.stage else 'STAGE: N/A'
        cv2.putText(frame, stage_text, (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        current_y += line_height

        # Angle (Smoothed Angle)
        if elbow_angle > 0:
            cv2.putText(frame, f'ELBOW ANGLE: {int(elbow_angle)}', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            current_y += line_height

        # Average Speed
        if self.avg_speed > 0:
            cv2.putText(frame, f'AVG SPEED: {self.avg_speed:.2f}s', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            current_y += line_height

        # --- System Ready Status ---
        # Positioned below Avg Speed or where it would be
        ready_y = current_y
        if not self.system_ready:
            cv2.putText(frame, 'GET IN PLANK POSITION', (info_x, ready_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            stability_percent = (self.stable_frame_count / self.stable_frames_required) * 100
            cv2.putText(frame, f'Stabilizing: {int(stability_percent)}%', (info_x, ready_y + int(line_height * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        else:
             cv2.putText(frame, 'READY', (info_x, ready_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        # --- Top Right Info Panel ---
        status_x = int(w * 0.65)
        status_y = int(h * 0.05)

        cv2.putText(frame, 'PUSH-UP COUNTER', (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Form Feedback (Only if ready and in DOWN stage)
        if self.system_ready and self.stage == "DOWN":
            feedback_y = status_y + line_height
            feedback_text = ""
            feedback_color = (0, 165, 255) # Orange for warning

            if not shoulder_alignment_ok:
                feedback_text = 'TIP: Keep shoulders level'
            elif elbow_angle > self.angle_threshold_down + 5: # Give a small buffer
                 feedback_text = 'TIP: Go deeper'
            elif elbow_angle < self.quality_depth_threshold:
                 feedback_text = 'Good depth!'
                 feedback_color = (0, 255, 0) # Green for good

            if feedback_text:
                 cv2.putText(frame, feedback_text, (status_x, feedback_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2, cv2.LINE_AA)


    def process_motion_frame(self, frame):
        """Processes a single frame using simple Motion Detection."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = blur
            print("Initializing background for motion detection...")
            # Display initialization text
            cv2.putText(frame, 'Initializing Motion Detection...', (int(w*0.1), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            return frame

        frame_delta = cv2.absdiff(self.background, blur)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_pixels = cv2.countNonZero(thresh)

        # Update background slowly
        self.background = cv2.addWeighted(self.background, 0.95, blur, 0.05, 0)

        current_time = time.time()

        if motion_pixels > self.motion_threshold:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0 # Reset if motion stops

        # Count if enough consecutive motion frames and cooldown passed
        if (self.consecutive_motion_frames >= self.motion_frames_required and
            current_time - self.last_motion_time > self.motion_cooldown):
            self.counter += 1
            self.last_motion_time = current_time
            print(f"Push-up #{self.counter} (Motion Detected)")
            # Reset consecutive count immediately after counting
            self.consecutive_motion_frames = 0

        # Display minimal UI for motion mode
        info_x = int(w * 0.02)
        info_y_start = int(h * 0.08)
        line_height = int(h * 0.06)
        cv2.putText(frame, f'PUSH-UPS: {self.counter}', (info_x, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Motion: {motion_pixels}', (info_x, info_y_start + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'MOTION DETECTION MODE', (info_x, info_y_start + 2*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        return frame

    def process_frame(self, frame):
        """Routes frame processing based on detection mode."""
        if self.detection_mode == "mediapipe":
            return self.process_mediapipe_frame(frame)
        else:
            return self.process_motion_frame(frame)

    def toggle_fullscreen(self, window_name):
        """Toggles the display window between fullscreen and normal."""
        self.full_screen = not self.full_screen
        if self.full_screen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("🖥️ Fullscreen mode enabled")
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            # You might need to adjust the resize dimensions if needed
            cv2.resizeWindow(window_name, 1200, 800)
            print("🖥️ Windowed mode enabled")

    def run(self):
        """Starts the camera capture and processing loop."""
        cap = cv2.VideoCapture(0) # 0 for default webcam
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        # Try to set a preferred resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30) # Request 30 FPS

        window_name = 'Final Balanced Push-Up Counter'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) # Allow resizing
        cv2.resizeWindow(window_name, 1200, 800) # Initial size

        print("\n" + "=" * 70)
        print("🎯 FINAL BALANCED PUSH-UP COUNTER")
        print("=" * 70)
        print("CAMERA: Face camera directly.")
        print("POSITION: Get into plank pose and hold to start.")
        print("\n✅ SETTINGS:")
        print(f"   - Ready Frames: {self.stable_frames_required}")
        print(f"   - UP Angle > {self.angle_threshold_up}° | DOWN Angle < {self.angle_threshold_down}°")
        print(f"   - Ready Check Angle > {self.angle_threshold_up_ready}°") # Added ready angle info
        print(f"   - Consecutive Frames: {self.consecutive_frames_required}")
        print(f"   - Smoothing Buffer: {self.smoothing_buffer_size}")
        print("\n🎮 CONTROLS:")
        print("   - 'q': Quit")
        print("   - 'r': Reset Counter")
        print("   - 'f': Toggle Fullscreen")
        print("   - 's': Save Stats (Print Summary)")
        print("=" * 70 + "\n")

        start_time = time.time()
        frame_count = 0
        key = 0 # Initialize key

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Attempt to reopen camera if reading fails
                    print("⚠️ Warning: Could not read frame. Attempting to reopen camera...")
                    cap.release()
                    time.sleep(1) # Wait a second before retrying
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("❌ Error: Failed to reopen camera. Exiting.")
                        break
                    else:
                        print("✅ Camera reopened successfully.")
                        continue # Skip the rest of the loop for this iteration


                frame_count += 1
                # Flip frame horizontally for a natural mirror view
                frame = cv2.flip(frame, 1)

                # Process the frame
                processed_frame = self.process_frame(frame)

                # --- Draw FPS and Instructions (Always Visible) ---
                h, w = processed_frame.shape[:2]
                current_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                cv2.putText(processed_frame, f'FPS: {current_fps:.1f}',
                            (w - int(w * 0.15), int(h * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                # Controls text at the bottom left
                controls_text = 'Q:Quit R:Reset F:Full S:Save'
                (text_width, text_height), baseline = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(processed_frame, controls_text,
                            (int(w * 0.02), h - int(h * 0.03) - baseline + text_height//2 ), # Position adjusted for baseline
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Display the frame
                cv2.imshow(window_name, processed_frame)

                # --- Handle Keyboard Input ---
                key = cv2.waitKey(1) & 0xFF # Use waitKey(1) for ~30+ fps
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("🔄 Counter reset! Get back into plank position.")
                    self.counter = 0
                    self.good_reps = 0
                    self.bad_reps = 0
                    self.speeds.clear()
                    self.avg_speed = 0
                    self.system_ready = False # Reset ready state
                    self.stable_frame_count = 0
                    self.stage = None # Reset current stage
                    self.left_elbow_buffer.clear() # Clear buffers
                    self.right_elbow_buffer.clear()
                    self.shoulder_buffer.clear()
                    # Reset time for FPS calculation
                    start_time = time.time()
                    frame_count = 0
                elif key == ord('f'):
                    self.toggle_fullscreen(window_name)
                elif key == ord('s'):
                    print("\n💾 Saving Workout Stats (Printing to Console)...")
                    # Use current elapsed time for summary
                    self.print_summary(time.time() - start_time, frame_count)


        except Exception as e:
            print(f"❌ An error occurred during execution: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
        finally:
            print("\nReleasing camera and closing windows...")
            cap.release()
            cv2.destroyAllWindows()
            # Attempt to close any lingering windows (important on some systems)
            for i in range(5):
                cv2.waitKey(1)

            # Print final summary if user quit normally
            if key == ord('q'):
                 self.print_summary(time.time() - start_time, frame_count)
            else:
                 print("\nWorkout ended unexpectedly or by closing the window.")


    def print_summary(self, total_time, total_frames):
        """Prints the workout summary to the console."""
        print("\n" + "=" * 70)
        print("🎉 PUSH-UP WORKOUT SUMMARY")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🏆 Total push-ups: {self.counter}")
        if self.counter > 0: # Avoid division by zero
            # Note: Quality stats are only calculated if reps were completed
            print(f"✅ Good form reps: {self.good_reps}")
            print(f"⚠️  Improvable reps: {self.bad_reps}")
            try:
                 accuracy = (self.good_reps / self.counter) * 100
                 print(f"🎯 Good form rate: {accuracy:.1f}%")
            except ZeroDivisionError:
                 print("🎯 Good form rate: N/A")

            if self.avg_speed > 0:
                print(f"⚡ Average speed: {self.avg_speed:.2f}s per rep")
        print(f"📈 Total frames processed: {total_frames}")
        avg_fps = total_frames / total_time if total_time > 0 else 0
        print(f"⏱️ Average FPS: {avg_fps:.1f}")
        print("=" * 70 + "\n")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        pushup_counter = FinalBalancedPushUpCounter()
        pushup_counter.run()
    except Exception as e:
        print(f"❌ Failed to start counter: {e}")
        print("Make sure you have necessary libraries installed:")
        print("pip install numpy opencv-python mediapipe")

