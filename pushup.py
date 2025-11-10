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
        self.last_stage = "UP" 
        self.rep_start_time = None
        self.last_rep_time = 0

        # --- TUNING PARAMETERS ---
        # Ready state
        self.system_ready = False
        self.stable_frames_required = 30  
        self.stable_frame_count = 0
        self.min_rep_interval = 0.8

        # Rep detection
        self.consecutive_frames_required = 3 
        self.consecutive_down_frames = 0
        self.consecutive_up_frames = 0

        # Angles
        self.angle_threshold_up_ready = 165 
        self.angle_threshold_up = 155     
        self.angle_threshold_down = 110   

        # Smoothing
        self.smoothing_buffer_size = 8
        self.left_elbow_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.right_elbow_buffer = deque(maxlen=self.smoothing_buffer_size) 
        self.shoulder_buffer = deque(maxlen=self.smoothing_buffer_size)

        # Confidence
        self.min_detection_confidence = 0.8 
        self.min_tracking_confidence = 0.8

        # --- NEW PLANK POSTURE CHECK ---
        # Max vertical distance (in pixels) between shoulders and hips to be considered a plank
        self.plank_max_y_diff = 100  # Tunable: 100 pixels is a good start for 720p

        # Quality Check
        self.quality_depth_threshold = 95
        self.quality_speed_min = 0.6 
        self.quality_speed_max = 4.0
        self.quality_shoulder_alignment_threshold = 35

        # Performance metrics
        self.good_reps = 0
        self.bad_reps = 0
        self.avg_speed = 0
        self.speeds = []

        # UI
        self.full_screen = False

        if MEDIAPIPE_AVAILABLE:
            self.setup_mediapipe()
        else:
            self.setup_motion_detection()

    def setup_mediapipe(self):
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
        print("Using MediaPipe Pose Detection - FINAL BALANCED FRONT VIEW PUSH-UPS")

    def setup_motion_detection(self):
        self.background = None
        self.motion_threshold = 5000
        self.last_motion_time = 0
        self.motion_cooldown = 1.5
        self.consecutive_motion_frames = 0
        self.motion_frames_required = 3
        self.detection_mode = "motion"
        print("Using Motion Detection - FINAL BALANCED FRONT VIEW PUSH-UPS")

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_shoulder_alignment(self, left_shoulder, right_shoulder, left_hip, right_hip):
        if not all(isinstance(p, (list, np.ndarray)) and len(p) == 2 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return False
        shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_y_diff = abs(left_hip[1] - right_hip[1])
        return shoulder_y_diff < self.quality_shoulder_alignment_threshold and \
               hip_y_diff < self.quality_shoulder_alignment_threshold

    def smooth_value(self, buffer, new_value):
        buffer.append(new_value)
        if buffer:
            return np.mean(buffer)
        return new_value

    def detect_pushup_quality(self, min_elbow_angle_during_rep, shoulder_alignment_ok, rep_time):
        depth_ok = min_elbow_angle_during_rep < self.quality_depth_threshold
        form_ok = shoulder_alignment_ok
        speed_ok = self.quality_speed_min < rep_time < self.quality_speed_max
        return depth_ok and form_ok and speed_ok

    def process_mediapipe_frame(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)

        smooth_left_angle = 0
        smooth_right_angle = 0
        shoulder_alignment_ok = False
        is_plank_posture = False  # Assume not in plank until proven

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                lm = self.mp_pose.PoseLandmark
                
                left_shoulder = [landmarks[lm.LEFT_SHOULDER.value].x * w, landmarks[lm.LEFT_SHOULDER.value].y * h]
                left_elbow = [landmarks[lm.LEFT_ELBOW.value].x * w, landmarks[lm.LEFT_ELBOW.value].y * h]
                left_wrist = [landmarks[lm.LEFT_WRIST.value].x * w, landmarks[lm.LEFT_WRIST.value].y * h]
                right_shoulder = [landmarks[lm.RIGHT_SHOULDER.value].x * w, landmarks[lm.RIGHT_SHOULDER.value].y * h]
                right_elbow = [landmarks[lm.RIGHT_ELBOW.value].x * w, landmarks[lm.RIGHT_ELBOW.value].y * h]
                right_wrist = [landmarks[lm.RIGHT_WRIST.value].x * w, landmarks[lm.RIGHT_WRIST.value].y * h]
                left_hip = [landmarks[lm.LEFT_HIP.value].x * w, landmarks[lm.LEFT_HIP.value].y * h]
                right_hip = [landmarks[lm.RIGHT_HIP.value].x * w, landmarks[lm.RIGHT_HIP.value].y * h]

                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                smooth_left_angle = self.smooth_value(self.left_elbow_buffer, left_elbow_angle)
                smooth_right_angle = self.smooth_value(self.right_elbow_buffer, right_elbow_angle)

                shoulder_alignment_ok = self.calculate_shoulder_alignment(left_shoulder, right_shoulder, left_hip, right_hip)
                
                # --- NEW PLANK CHECK ---
                avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                body_vertical_diff = abs(avg_shoulder_y - avg_hip_y)
                is_plank_posture = body_vertical_diff < self.plank_max_y_diff

                current_time = time.time()

                # --- "Ready" State (Must have BOTH arms straight AND be in a plank) ---
                if not self.system_ready:
                    if (smooth_left_angle > self.angle_threshold_up_ready and
                        smooth_right_angle > self.angle_threshold_up_ready and
                        is_plank_posture): 
                        
                        self.stable_frame_count = min(self.stable_frame_count + 1, self.stable_frames_required)
                        if self.stable_frame_count >= self.stable_frames_required:
                            self.system_ready = True
                            self.stage = "UP"
                            self.last_stage = "UP"
                            print("✅ System Ready - Start Push-ups!")
                    else:
                        self.stable_frame_count = max(0, self.stable_frame_count - 1)

                # --- Counting Logic ---
                if self.system_ready:
                    # Reset if plank is lost
                    if not is_plank_posture:
                        self.stage = "UP"
                        self.consecutive_down_frames = 0
                        self.consecutive_up_frames = 0
                        self.rep_start_time = None
                    
                    # If in a valid plank, run counting logic
                    else:
                        current_stage_potential = self.stage
                        
                        if smooth_left_angle > self.angle_threshold_up and smooth_right_angle > self.angle_threshold_up:
                            self.consecutive_up_frames += 1
                            self.consecutive_down_frames = 0
                            current_stage_potential = "UP"
                        elif smooth_left_angle < self.angle_threshold_down and smooth_right_angle < self.angle_threshold_down:
                            self.consecutive_down_frames += 1
                            self.consecutive_up_frames = 0
                            current_stage_potential = "DOWN"
                        else:
                            self.consecutive_up_frames = 0
                            self.consecutive_down_frames = 0

                        # --- Transition to UP (Rep complete) ---
                        if current_stage_potential == "UP" and self.consecutive_up_frames >= self.consecutive_frames_required:
                            if self.stage == "DOWN":
                                self.counter += 1
                                if self.rep_start_time:
                                    rep_time = current_time - self.rep_start_time
                                    if 0.3 < rep_time < 10.0:
                                        self.speeds.append(rep_time)
                                        self.avg_speed = np.mean(self.speeds[-10:]) if self.speeds else 0
                                        
                                        min_left = min(list(self.left_elbow_buffer)) if self.left_elbow_buffer else smooth_left_angle
                                        min_right = min(list(self.right_elbow_buffer)) if self.right_elbow_buffer else smooth_right_angle
                                        min_angle_this_rep = min(min_left, min_right)

                                        if self.detect_pushup_quality(min_angle_this_rep, shoulder_alignment_ok, rep_time):
                                            self.good_reps += 1
                                            print(f"✅ Rep #{self.counter} (Good) - Time: {rep_time:.2f}s")
                                        else:
                                            self.bad_reps += 1
                                            print(f"⚠️ Rep #{self.counter} (Bad) - Time: {rep_time:.2f}s")
                                    else:
                                         print(f"⚠️ Rep #{self.counter} (Ignored - Unrealistic Time: {rep_time:.2f}s)")
                                self.last_rep_time = current_time
                                self.rep_start_time = None
                            self.stage = "UP"

                        # --- Transition to DOWN (Rep start) ---
                        elif current_stage_potential == "DOWN" and self.consecutive_down_frames >= self.consecutive_frames_required:
                            # Can only start a rep if we were in UP and plank is held
                            if (self.stage == "UP" and 
                                (current_time - self.last_rep_time > self.min_rep_interval) and
                                is_plank_posture): # Redundant check, but good for safety
                                
                                self.rep_start_time = current_time
                                self.left_elbow_buffer.clear()
                                self.right_elbow_buffer.clear()
                                print(f"🏋️ Rep #{self.counter + 1} - Going down...")
                                self.stage = "DOWN"

                # --- DRAWING ---
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
                )

                cv2.putText(frame, f"{int(left_elbow_angle)}", tuple(np.add(left_elbow, [10, -10]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(right_elbow_angle)}", tuple(np.add(right_elbow, [-40, -10]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                display_angle = (smooth_left_angle + smooth_right_angle) / 2
                self.display_pushup_info(frame, display_angle, shoulder_alignment_ok, is_plank_posture)

            else:
                # NO POSE DETECTED
                self.system_ready = False
                self.stable_frame_count = 0
                self.stage = None
                cv2.putText(frame, 'NO POSE DETECTED', (int(w*0.1), int(h*0.1)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Face camera directly', (int(w*0.1), int(h*0.15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing frame: {e}")

        return frame

    def display_pushup_info(self, frame, elbow_angle, shoulder_alignment_ok, is_plank_posture):
        h, w = frame.shape[:2]
        info_x = int(w * 0.02)
        info_y_start = int(h * 0.08)
        line_height = int(h * 0.05)

        cv2.putText(frame, f'PUSH-UPS: {self.counter}', (info_x, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        current_y = info_y_start + int(line_height * 1.4)

        stage_text = f'STAGE: {self.stage}' if self.stage else 'STAGE: N/A'
        cv2.putText(frame, stage_text, (info_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        current_y += line_height

        if elbow_angle > 0:
            cv2.putText(frame, f'AVG ELBOW ANGLE: {int(elbow_angle)}', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            current_y += line_height

        if self.avg_speed > 0:
            cv2.putText(frame, f'AVG SPEED: {self.avg_speed:.2f}s', (info_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            current_y += line_height

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

        status_x = int(w * 0.65)
        status_y = int(h * 0.05)
        cv2.putText(frame, 'PUSH-UP COUNTER', (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # --- NEW CONSOLIDATED FEEDBACK ---
        feedback_y = status_y + line_height
        feedback_text = ""
        feedback_color = (0, 165, 255) # Orange warning
        
        if self.system_ready:
            if not is_plank_posture:
                feedback_text = 'TIP: Straighten body (plank)'
                feedback_color = (0, 0, 255) # Red
            elif not shoulder_alignment_ok:
                feedback_text = 'TIP: Keep shoulders level'
            elif self.stage == "DOWN" and elbow_angle > self.angle_threshold_down + 5:
                 feedback_text = 'TIP: Go deeper'
            elif self.stage == "DOWN" and elbow_angle < self.quality_depth_threshold:
                 feedback_text = 'Good depth!'
                 feedback_color = (0, 255, 0) # Green
        
        if feedback_text:
             cv2.putText(frame, feedback_text, (status_x, feedback_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2, cv2.LINE_AA)


    def process_motion_frame(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.background is None:
            self.background = blur
            return frame
        frame_delta = cv2.absdiff(self.background, blur)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_pixels = cv2.countNonZero(thresh)
        self.background = cv2.addWeighted(self.background, 0.95, blur, 0.05, 0)
        if motion_pixels > self.motion_threshold:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0
        if (self.consecutive_motion_frames >= self.motion_frames_required and
            time.time() - self.last_motion_time > self.motion_cooldown):
            self.counter += 1
            self.last_motion_time = time.time()
            self.consecutive_motion_frames = 0
        cv2.putText(frame, f'PUSH-UPS: {self.counter} (Motion)', (int(w*0.02), int(h*0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def process_frame(self, frame):
        if self.detection_mode == "mediapipe":
            return self.process_mediapipe_frame(frame)
        return self.process_motion_frame(frame)

    def toggle_fullscreen(self, window_name):
        self.full_screen = not self.full_screen
        if self.full_screen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        window_name = 'Final Balanced Push-Up Counter'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            cv2.imshow(window_name, processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                print("🔄 Counter reset! Get back into plank position.")
                self.counter = 0
                self.good_reps = 0
                self.bad_reps = 0
                self.system_ready = False
                self.stable_frame_count = 0
                self.stage = None
                self.speeds.clear()
                self.avg_speed = 0
                self.left_elbow_buffer.clear()
                self.right_elbow_buffer.clear()
                # Reset time for FPS calculation
                start_time = time.time()
                frame_count = 0
            elif key == ord('f'): self.toggle_fullscreen(window_name)
            elif key == ord('s'): self.print_summary(time.time() - start_time, frame_count)

        # Print summary on quit
        self.print_summary(time.time() - start_time, frame_count)
        
        cap.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1) # Close lingering windows

    def print_summary(self, total_time, total_frames):
        print("\n" + "="*30 + " SUMMARY " + "="*30)
        print(f"Total Push-ups: {self.counter}")
        print(f"Good Form: {self.good_reps} | Needs Improvement: {self.bad_reps}")
        if self.counter > 0:
            accuracy = (self.good_reps / self.counter) * 100
            print(f"Good Form Rate: {accuracy:.1f}%")
        if self.avg_speed > 0:
            print(f"Average Speed: {self.avg_speed:.2f}s per rep")
        print(f"Total Time: {total_time:.1f}s")
        print("="*70 + "\n")

if __name__ == "__main__":
    pushup_counter = FinalBalancedPushUpCounter()
    pushup_counter.run()