import cv2
import numpy as np
import dlib
import time
import threading


class EyeTracker:
    def __init__(self, video_src=1, predictor_path="C:/Users/asus/PycharmProjects/spekoach/shape_predictor_68_face_landmarks.dat", display=True):
        self.cap = cv2.VideoCapture(video_src)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.start_time = time.time()
        self.right_count = 0
        self.left_count = 0
        self.center_count = 0
        self.gaze_score_ratio = 0.0  # to store final gaze score
        # placeholders filled each frame
        self.frame = None
        self.gray = None
        self.running = True
        self.display = display

    def get_gaze_ratio(self, eye_points, facial_landmark):
        eye_region = np.array([
            (facial_landmark.part(eye_points[0]).x, facial_landmark.part(eye_points[0]).y),
            (facial_landmark.part(eye_points[1]).x, facial_landmark.part(eye_points[1]).y),
            (facial_landmark.part(eye_points[2]).x, facial_landmark.part(eye_points[2]).y),
            (facial_landmark.part(eye_points[3]).x, facial_landmark.part(eye_points[3]).y),
            (facial_landmark.part(eye_points[4]).x, facial_landmark.part(eye_points[4]).y),
            (facial_landmark.part(eye_points[5]).x, facial_landmark.part(eye_points[5]).y)
        ], np.int32)

        height, width = self.frame.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(self.gray, self.gray, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        # guard against invalid crop
        if min_x >= max_x or min_y >= max_y:
            return 0.0

        masked_eye = eye[min_y: max_y, min_x: max_x]
        if masked_eye.size == 0:
            return 0.0

        def thresh_cali(bright):
            if 90 < bright <= 150:
                return 40
            elif bright > 150:
                return 75
            else:
                return 30

        # initial rough threshold and adaptive recalibration
        thresh = 40
        _, threshold_eye = cv2.threshold(masked_eye, thresh, 255, cv2.THRESH_BINARY)
        brightness = np.mean(threshold_eye)

        thresh = thresh_cali(brightness)
        _, threshold_eye = cv2.threshold(masked_eye, thresh, 255, cv2.THRESH_BINARY)

        h, w = threshold_eye.shape
        left_side_threshold = threshold_eye[0: h, 0: int(w / 2)]
        right_side_threshold = threshold_eye[0: h, int(w / 2): w]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_white = cv2.countNonZero(right_side_threshold)

        if right_side_white == 0:
            gaze_ratio = float('inf')  # treat as extreme left if no white on right
        else:
            gaze_ratio = left_side_white / right_side_white
            
        return gaze_ratio

        
    def start_tracking(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            self.frame = frame
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.gray.dtype != np.uint8:
                self.gray = (self.gray * 255).astype(np.uint8)

            # Only detect if gray is valid
            if self.gray.ndim != 2 or self.gray.size == 0:
                continue

            faces = self.detector(self.gray)
            
            elapsed = time.time() - self.start_time

            for face in faces:
                landmarks = self.predictor(self.gray, face)
                gaze_ratio_left_eye = self.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = self.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

                # handle infinite ratio when one eye segment had zero white pixels
                if np.isinf(gaze_ratio_left_eye) and np.isinf(gaze_ratio_right_eye):
                    gaze_ratio = float('inf')
                else:
                    # replace inf with a large number so average works
                    l = gaze_ratio_left_eye if not np.isinf(gaze_ratio_left_eye) else 10.0
                    r = gaze_ratio_right_eye if not np.isinf(gaze_ratio_right_eye) else 10.0
                    gaze_ratio = (l + r) / 2

                if gaze_ratio < 0.8:
                    cv2.putText(frame, "Don't have eye contact", (50, 150), self.font, 3, (255, 0, 0))
                    self.right_count += 1
                elif 0.8 <= gaze_ratio < 2:
                    cv2.putText(frame, "Center", (50, 150), self.font, 3, (255, 0, 0))
                    self.center_count += 1
                else:
                    cv2.putText(frame, "Don't have eye contact", (50, 150), self.font, 2, (255, 0, 0))
                    self.left_count += 1

            cv2.imshow("frame", frame)


            if self.display:
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            else:
               # when running headless (no GUI) yield CPU and let external code stop tracker.running
                time.sleep(0.01)
            

        total_count = self.center_count + self.left_count + self.right_count
        if total_count == 0:
            self.gaze_score_ratio = 0.0
        else:
            self.gaze_score_ratio = (self.center_count / total_count) * 100

        self.cap.release()
        cv2.destroyAllWindows()

        self.running = False
