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

class FinalLungeCounter:
    def __init__(self):
        # Core state
        self.counter = 0
        self.stage = None
        self.last_stage = "UP"
        self.rep_start_time = None
        self.last_rep_time = 0
        self.current_leg = "N/A" # Will be detected

        # --- TUNING PARAMETERS ---
        # Ready state: User must hold standing pose for ~0.6 seconds
        self.system_ready = False
        self.stable_frames_required = 18 # Hold standing pose for 18 frames
        self.stable_frame_count = 0
        self.min_rep_interval = 1.2 # Slightly longer interval for lunges

        # Rep detection: Need 2 consecutive frames in a state
        self.consecutive_frames_required = 2
        self.consecutive_down_frames = 0
        self.consecutive_up_frames = 0

        # Angles: Define UP and DOWN states (more relaxed than squats)
        self.angle_threshold_up = 155 # Angle > 155 degrees is UP (both knees)
        self.angle_threshold_down = 135 # Angle < 135 degrees is DOWN (both knees)

        # Smoothing: Use mean for responsiveness, moderate buffer size
        self.smoothing_buffer_size = 8
        self.front_knee_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.back_knee_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.hip_balance_buffer = deque(maxlen=self.smoothing_buffer_size) # For balance metric

        # Confidence: MediaPipe detection confidence
        self.min_detection_confidence = 0.65
        self.min_tracking_confidence = 0.65

        # Quality Check (for summary, not displayed in real-time)
        self.quality_front_knee_min = 75
        self.quality_front_knee_max = 105
        self.quality_back_knee_max = 120 # Back knee needs good bend
        self.quality_balance_threshold = 55 # Max pixel offset for balance
        self.quality_speed_min = 1.2
        self.quality_speed_max = 6.0

        # Performance metrics
        self.good_reps = 0
        self.bad_reps = 0
        self.avg_speed = 0
        self.speeds = []
        self.balance_history = [] # Track balance during the DOWN phase

        # UI
        self.full_screen = False
        self.window_width = 1200
        self.window_height = 800
        self.current_scale = 1.0
        self.font_scale = 1.0
        self.text_thickness = 2

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
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detection_mode = "mediapipe"
        print("Using MediaPipe Pose Detection - FINAL SIDE VIEW LUNGES")

    def setup_motion_detection(self):
        """Initializes Motion Detection fallback."""
        self.background = None
        self.motion_threshold = 10000 # Higher threshold for larger lunge movements
        self.last_motion_time = 0
        self.motion_cooldown = 2.5 # Longer cooldown
        self.consecutive_motion_frames = 0
        self.motion_frames_required = 3
        self.detection_mode = "motion"
        print("Using Motion Detection - FINAL SIDE VIEW LUNGES")

    def calculate_angle(self, a, b, c):
        """Calculates the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_balance(self, front_ankle, back_ankle, hip):
        """Estimates balance based on hip X-position relative to ankle X-positions."""
        # Ensure points are valid lists/tuples with 2 elements
        if not (isinstance(front_ankle, (list, tuple)) and len(front_ankle) >= 1 and
                isinstance(back_ankle, (list, tuple)) and len(back_ankle) >= 1 and
                isinstance(hip, (list, tuple)) and len(hip) >= 1):
            return 999 # Return a high offset if points are invalid

        hip_x = hip[0]
        front_ankle_x = front_ankle[0]
        back_ankle_x = back_ankle[0]

        # Avoid division by zero if ankles overlap (unlikely in side view)
        if abs(front_ankle_x - back_ankle_x) < 1:
            return abs(hip_x - front_ankle_x) # Offset from one ankle

        center = (front_ankle_x + back_ankle_x) / 2
        balance_offset = abs(hip_x - center)
        return balance_offset

    def detect_leading_leg(self, left_ankle, right_ankle):
        """Determines the forward leg based on X-coordinate in side view."""
        # Assumes camera is positioned such that the forward leg has a larger X value
        # Adjust if your camera setup is different
        if not (isinstance(left_ankle, (list, tuple)) and len(left_ankle) >= 1 and
                isinstance(right_ankle, (list, tuple)) and len(right_ankle) >= 1):
             return "N/A" # Cannot determine if points invalid

        if left_ankle[0] > right_ankle[0]:
            return "LEFT"
        elif right_ankle[0] > left_ankle[0]:
            return "RIGHT"
        else:
            return "N/A" # Ankles aligned?

    def smooth_value(self, buffer, new_value):
        """Applies simple averaging smoothing."""
        buffer.append(new_value)
        return np.mean(buffer)

    def detect_lunge_quality(self, min_front_knee_angle, min_back_knee_angle, avg_balance_offset, rep_time):
        """Determines if the completed lunge met quality criteria."""
        front_knee_ok = self.quality_front_knee_min < min_front_knee_angle < self.quality_front_knee_max
        back_knee_ok = min_back_knee_angle < self.quality_back_knee_max
        balance_ok = avg_balance_offset < self.quality_balance_threshold
        speed_ok = self.quality_speed_min < rep_time < self.quality_speed_max
        return front_knee_ok and back_knee_ok and balance_ok and speed_ok

    # --- Responsive UI Functions ---
    def update_scale_factors(self, frame_width, frame_height):
        """Adjusts UI scaling based on window size."""
        base_width = 1280
        base_height = 720
        width_scale = frame_width / base_width
        height_scale = frame_height / base_height
        self.current_scale = min(width_scale, height_scale, 1.5)
        self.font_scale = max(0.5, min(2.0, self.current_scale))
        self.text_thickness = max(1, int(2 * self.current_scale))

    def get_scaled_font_properties(self):
        """Returns scaled font sizes and thicknesses."""
        return {
            'scale_main': self.font_scale * 1.2,
            'scale_large': self.font_scale * 0.9,
            'scale_medium': self.font_scale * 0.7,
            'scale_small': self.font_scale * 0.5,
            'thickness_main': max(2, self.text_thickness + 1),
            'thickness_normal': self.text_thickness,
            'thickness_small': max(1, self.text_thickness - 1)
        }

    def process_mediapipe_frame(self, frame):
        """Processes a single frame using MediaPipe for lunge detection."""
        h, w = frame.shape[:2]
        self.update_scale_factors(w, h)
        font_props = self.get_scaled_font_properties()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        front_knee_angle = 0
        back_knee_angle = 0
        balance = 999 # High value indicates poor balance initially

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Key points
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
                left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
                right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
                right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

                # Determine leading leg
                self.current_leg = self.detect_leading_leg(left_ankle, right_ankle)

                # Calculate angles and balance based on leading leg
                raw_front_knee = 0
                raw_back_knee = 0
                raw_balance = 999

                if self.current_leg == "LEFT":
                    raw_front_knee = self.calculate_angle(left_hip, left_knee, left_ankle)
                    raw_back_knee = self.calculate_angle(right_hip, right_knee, right_ankle)
                    raw_balance = self.calculate_balance(left_ankle, right_ankle, left_hip)
                elif self.current_leg == "RIGHT":
                    raw_front_knee = self.calculate_angle(right_hip, right_knee, right_ankle)
                    raw_back_knee = self.calculate_angle(left_hip, left_knee, left_ankle)
                    raw_balance = self.calculate_balance(right_ankle, left_ankle, right_hip)

                # Apply smoothing
                front_knee_angle = self.smooth_value(self.front_knee_buffer, raw_front_knee)
                back_knee_angle = self.smooth_value(self.back_knee_buffer, raw_back_knee)
                balance = self.smooth_value(self.hip_balance_buffer, raw_balance)


                current_time = time.time()

                # --- READY STATE LOGIC ---
                if not self.system_ready:
                    if front_knee_angle > self.angle_threshold_up and back_knee_angle > self.angle_threshold_up:
                        self.stable_frame_count = min(self.stable_frame_count + 1, self.stable_frames_required)
                        if self.stable_frame_count >= self.stable_frames_required:
                            self.system_ready = True
                            self.stage = "UP"
                            self.last_stage = "UP"
                            print("✅ System Ready - Start Lunges!")
                    else:
                        self.stable_frame_count = max(0, self.stable_frame_count - 1)

                # --- COUNTING LOGIC (Only if ready) ---
                if self.system_ready:
                    # Determine potential stage
                    if front_knee_angle > self.angle_threshold_up and back_knee_angle > self.angle_threshold_up:
                        self.consecutive_up_frames += 1
                        self.consecutive_down_frames = 0
                        current_stage_potential = "UP"
                    elif front_knee_angle < self.angle_threshold_down and back_knee_angle < self.angle_threshold_down:
                        self.consecutive_down_frames += 1
                        self.consecutive_up_frames = 0
                        current_stage_potential = "DOWN"
                    else:
                        self.consecutive_up_frames = 0
                        self.consecutive_down_frames = 0
                        current_stage_potential = self.stage

                    # Confirm stage change
                    if current_stage_potential == "UP" and self.consecutive_up_frames >= self.consecutive_frames_required:
                        if self.stage == "DOWN":
                            self.counter += 1
                            if self.rep_start_time:
                                rep_time = current_time - self.rep_start_time
                                self.speeds.append(rep_time)
                                self.avg_speed = np.mean(self.speeds[-10:]) if self.speeds else 0

                                min_front = min(self.front_knee_buffer) if self.front_knee_buffer else front_knee_angle
                                min_back = min(self.back_knee_buffer) if self.back_knee_buffer else back_knee_angle
                                avg_bal = np.mean(self.balance_history) if self.balance_history else balance

                                if self.detect_lunge_quality(min_front, min_back, avg_bal, rep_time):
                                    self.good_reps += 1
                                    print(f"✅ Rep #{self.counter} (Good) - {self.current_leg} - Time: {rep_time:.2f}s")
                                else:
                                    self.bad_reps += 1
                                    print(f"⚠️ Rep #{self.counter} (Bad) - {self.current_leg} - Time: {rep_time:.2f}s")
                                self.last_rep_time = current_time
                                self.balance_history.clear() # Reset balance history for next rep
                            self.rep_start_time = None
                        self.stage = "UP"

                    elif current_stage_potential == "DOWN" and self.consecutive_down_frames >= self.consecutive_frames_required:
                        if self.stage == "UP":
                            self.rep_start_time = current_time
                            self.front_knee_buffer.clear()
                            self.back_knee_buffer.clear()
                            self.balance_history.clear() # Start collecting balance data
                            print(f"🏋️ Rep #{self.counter + 1} - {self.current_leg} leg forward - Going down...")
                        self.stage = "DOWN"

                    # Collect balance data only when confirmed in DOWN state
                    if self.stage == "DOWN":
                        self.balance_history.append(raw_balance) # Store raw balance during DOWN

                # --- DRAWING ---
                landmark_radius = max(2, int(3 * self.current_scale))
                landmark_thickness = max(1, int(2 * self.current_scale))
                connection_thickness = max(1, int(2 * self.current_scale))

                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=landmark_thickness, circle_radius=landmark_radius),
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=connection_thickness, circle_radius=landmark_radius)
                )

                angle_text_size = max(0.3, 0.5 * self.current_scale)
                angle_thickness = max(1, self.text_thickness - 1)
                # Draw angles on the correct knees based on leading leg
                if self.current_leg == "LEFT":
                     cv2.putText(frame, f'F:{int(raw_front_knee)}', tuple(np.multiply(left_knee, [1, 1]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, angle_text_size, (255, 255, 0), angle_thickness)
                     cv2.putText(frame, f'B:{int(raw_back_knee)}', tuple(np.multiply(right_knee, [1, 1]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, angle_text_size, (255, 150, 150), angle_thickness)
                elif self.current_leg == "RIGHT":
                     cv2.putText(frame, f'F:{int(raw_front_knee)}', tuple(np.multiply(right_knee, [1, 1]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, angle_text_size, (255, 255, 0), angle_thickness)
                     cv2.putText(frame, f'B:{int(raw_back_knee)}', tuple(np.multiply(left_knee, [1, 1]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, angle_text_size, (255, 150, 150), angle_thickness)


                self.display_lunge_info(frame, front_knee_angle, back_knee_angle, balance, font_props)

            else:
                # --- NO POSE DETECTED ---
                self.system_ready = False
                self.stable_frame_count = 0
                self.stage = None
                cv2.putText(frame, 'NO POSE DETECTED',
                            (int(w * 0.1), int(h * 0.1)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_main'], (0, 0, 255), font_props['thickness_main'])
                cv2.putText(frame, 'Show side profile to camera',
                            (int(w * 0.1), int(h * 0.15)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
                # No call to display_lunge_info

        except Exception as e:
            print(f"Error processing frame: {e}")
            self.display_lunge_info(frame, 0, 0, 999, font_props) # Show basic UI

        return frame

    def display_lunge_info(self, frame, front_knee_angle, back_knee_angle, balance, font_props):
        """Displays the UI elements on the frame."""
        h, w = frame.shape[:2]
        line_height = int(h * 0.05)

        # --- Top Left Info Panel ---
        info_x = int(w * 0.02)
        info_y_start = int(h * 0.08)
        current_y = info_y_start

        # Counter
        cv2.putText(frame, f'LUNGES: {self.counter}', (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_main'], (0, 255, 0), font_props['thickness_main'])
        current_y += int(line_height * 1.5)

        # Stage & Leading Leg
        cv2.putText(frame, f'STAGE: {self.stage}', (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_large'], (255, 255, 255), font_props['thickness_normal'])
        current_y += line_height
        cv2.putText(frame, f'LEADING: {self.current_leg}', (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
        current_y += line_height

        # Angles
        if front_knee_angle > 0:
             cv2.putText(frame, f'FRONT KNEE: {int(front_knee_angle)}', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
             current_y += line_height
        if back_knee_angle > 0:
             cv2.putText(frame, f'BACK KNEE: {int(back_knee_angle)}', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
             current_y += line_height

        # Balance
        cv2.putText(frame, f'BALANCE OFFSET: {int(balance)}', (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
        current_y += line_height

        # Average Speed
        if self.avg_speed > 0:
            cv2.putText(frame, f'AVG SPEED: {self.avg_speed:.2f}s', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 0), font_props['thickness_normal'])
            current_y += line_height

        # --- System Ready Status ---
        if not self.system_ready:
            cv2.putText(frame, 'GET IN STANDING POSITION', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (0, 0, 255), font_props['thickness_normal'])
            stability_percent = (self.stable_frame_count / self.stable_frames_required) * 100
            cv2.putText(frame, f'Stabilizing: {int(stability_percent)}%', (info_x, current_y + int(line_height * 0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (0, 165, 255), font_props['thickness_small'])
        else:
             cv2.putText(frame, 'READY', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (0, 255, 0), font_props['thickness_normal'])

        # --- Top Right Info Panel ---
        status_x = int(w * 0.65)
        status_y = int(h * 0.05)
        cv2.putText(frame, 'LUNGE COUNTER', (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (0, 255, 255), font_props['thickness_normal'])

        # Form Feedback
        if self.system_ready and self.stage == "DOWN":
            feedback_y = status_y + line_height
            if not (self.quality_front_knee_min < front_knee_angle < self.quality_front_knee_max):
                 cv2.putText(frame, f'TIP: Adjust front knee (aim {self.quality_front_knee_min}-{self.quality_front_knee_max})', (status_x, feedback_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (0, 165, 255), font_props['thickness_small'])
            elif back_knee_angle > self.quality_back_knee_max:
                 cv2.putText(frame, 'TIP: Bend back knee more', (status_x, feedback_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (0, 165, 255), font_props['thickness_small'])
            elif balance > self.quality_balance_threshold:
                 cv2.putText(frame, 'TIP: Improve balance', (status_x, feedback_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (0, 165, 255), font_props['thickness_small'])
            else:
                 cv2.putText(frame, 'Good form!', (status_x, feedback_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (0, 255, 0), font_props['thickness_small'])


    def process_motion_frame(self, frame):
        """Processes a single frame using simple Motion Detection."""
        h, w = frame.shape[:2]
        self.update_scale_factors(w, h)
        font_props = self.get_scaled_font_properties()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = blur
            print("Initializing background for motion detection...")
            return frame

        frame_delta = cv2.absdiff(self.background, blur)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_pixels = cv2.countNonZero(thresh)

        self.background = cv2.addWeighted(self.background, 0.95, blur, 0.05, 0)
        current_time = time.time()

        if motion_pixels > self.motion_threshold:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0

        if (self.consecutive_motion_frames >= self.motion_frames_required and
            current_time - self.last_motion_time > self.motion_cooldown):
            self.counter += 1
            self.last_motion_time = current_time
            print(f"Lunge #{self.counter} (Motion Detected)")
            self.consecutive_motion_frames = 0

        # Display minimal UI for motion mode
        info_x = int(w * 0.02)
        info_y_start = int(h * 0.08)
        line_height = int(h * 0.06)
        cv2.putText(frame, f'LUNGES: {self.counter}', (info_x, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_main'], (0, 255, 0), font_props['thickness_main'])
        cv2.putText(frame, f'Motion: {motion_pixels}', (info_x, info_y_start + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_large'], (255, 255, 255), font_props['thickness_normal'])
        cv2.putText(frame, 'MOTION DETECTION MODE', (info_x, info_y_start + 2*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (0, 165, 255), font_props['thickness_normal'])

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
            cv2.resizeWindow(window_name, self.window_width, self.window_height)
            print("🖥️ Windowed mode enabled")

    def handle_window_resize(self, window_name):
        """Updates internal window dimensions if the window is resized."""
        try:
            rect = cv2.getWindowImageRect(window_name)
            current_width, current_height = rect[2], rect[3]
            if current_width > 0 and current_height > 0:
                 if abs(current_width - self.window_width) > 1 or abs(current_height - self.window_height) > 1:
                    self.window_width = current_width
                    self.window_height = current_height
                    return True
        except cv2.error: pass
        return False

    def run(self):
        """Starts the camera capture and processing loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = 'Final Lunge Counter - Resize Window | F: Fullscreen'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        print("\n" + "=" * 70)
        print("🎯 FINAL LUNGE COUNTER")
        print("=" * 70)
        print("CAMERA: Place camera to your SIDE.")
        print("POSITION: Show full body profile. Stand still to begin.")
        print("\n✅ SETTINGS:")
        print(f"   - Ready Frames: {self.stable_frames_required}")
        print(f"   - UP Angle > {self.angle_threshold_up}° | DOWN Angle < {self.angle_threshold_down}°")
        print(f"   - Consecutive Frames: {self.consecutive_frames_required}")
        print(f"   - Smoothing Buffer: {self.smoothing_buffer_size}")
        print("\n🎮 CONTROLS:")
        print("   - 'q': Quit")
        print("   - 'r': Reset Counter")
        print("   - 'f': Toggle Fullscreen")
        print("   - 's': Save Stats (Print Summary)")
        print("   - Drag window edges to resize")
        print("=" * 70 + "\n")

        start_time = time.time()
        frame_count = 0
        last_resize_check = time.time()
        key = 0 # Initialize key

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame.")
                    break

                frame_count += 1
                # NO FLIP for side view
                processed_frame = self.process_frame(frame)

                # --- Draw FPS and Instructions ---
                h, w = processed_frame.shape[:2]
                font_props = self.get_scaled_font_properties()
                fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                cv2.putText(processed_frame, f'FPS: {fps:.1f}',
                            (w - int(w * 0.15), int(h * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_medium'], (255, 255, 255), font_props['thickness_normal'])
                cv2.putText(processed_frame, 'Q:Quit R:Reset F:Full S:Save',
                            (int(w * 0.02), h - int(h * 0.03)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_props['scale_small'], (255, 255, 255), font_props['thickness_small'])

                cv2.imshow(window_name, processed_frame)

                # --- Handle Window Resize ---
                current_time_check = time.time()
                if current_time_check - last_resize_check > 0.5:
                    self.handle_window_resize(window_name)
                    last_resize_check = current_time_check

                # --- Handle Keyboard Input ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("🔄 Counter reset! Get back into standing position.")
                    self.counter = 0
                    self.good_reps = 0
                    self.bad_reps = 0
                    self.speeds.clear()
                    self.avg_speed = 0
                    self.balance_history.clear()
                    self.system_ready = False
                    self.stable_frame_count = 0
                    self.stage = None
                    self.front_knee_buffer.clear()
                    self.back_knee_buffer.clear()
                    self.hip_balance_buffer.clear()
                    start_time = time.time() # Reset timer for FPS too
                    frame_count = 0
                elif key == ord('f'):
                    self.toggle_fullscreen(window_name)
                elif key == ord('s'):
                    print("\n💾 Saving Workout Stats (Printing to Console)...")
                    self.print_summary(time.time() - start_time, frame_count)

        except Exception as e:
            print(f"❌ An error occurred during execution: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nCamera released and windows closed.")
            if key == ord('q'): # Only print summary if user quit normally
                self.print_summary(time.time() - start_time, frame_count)

    def print_summary(self, total_time, total_frames):
        """Prints the workout summary to the console."""
        print("\n" + "=" * 70)
        print("🎉 LUNGE WORKOUT SUMMARY")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🏆 Total lunges: {self.counter}")
        if self.counter > 0:
            print(f"✅ Good form reps: {self.good_reps}")
            print(f"⚠️  Improvable reps: {self.bad_reps}")
            accuracy = (self.good_reps / self.counter) * 100
            print(f"🎯 Good form rate: {accuracy:.1f}%")
            if self.avg_speed > 0:
                print(f"⚡ Average speed: {self.avg_speed:.2f}s per rep")
        print(f"📈 Total frames processed: {total_frames}")
        print("=" * 70 + "\n")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        lunge_counter = FinalLungeCounter()
        lunge_counter.run()
    except Exception as e:
        print(f"❌ Failed to start counter: {e}")
        print("Make sure you have necessary libraries installed:")
        print("pip install numpy opencv-python mediapipe")
