from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class ActionDetector:
    """Detects face actions: movement, blinks, and smiles."""

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # State tracking
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_left_eye_ratio: Optional[float] = None
        self.prev_right_eye_ratio: Optional[float] = None
        self.blink_cooldown = 0
        
        # Movement thresholds
        self.movement_threshold = 20  # pixels
        self.blink_threshold = 0.2
        self.smile_threshold = 0.35
        
        # Eye landmark indices (MediaPipe 468 landmarks)
        self.LEFT_EYE_UPPER = [159, 145]
        self.LEFT_EYE_LOWER = [23, 130]
        self.RIGHT_EYE_UPPER = [386, 374]
        self.RIGHT_EYE_LOWER = [253, 359]
        
        # Mouth landmark indices
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14

    def detect_actions(
        self,
        frame: cv2.Mat,
        face_box: Tuple[int, int, int, int],
    ) -> List[str]:
        """
        Detect actions on a locked face.
        
        Args:
            frame: Current video frame
            face_box: Bounding box of the locked face
            
        Returns:
            List of detected action strings
        """
        actions = []
        
        # Get landmarks
        landmarks = self._get_landmarks(frame, face_box)
        if landmarks is None:
            self.blink_cooldown = max(0, self.blink_cooldown - 1)
            return actions
        
        # Detect movement
        movement = self._detect_movement(landmarks, frame.shape)
        if movement:
            actions.append(movement)
        
        # Detect blink
        if self._detect_blink(landmarks, frame.shape):
            actions.append("blink")
        
        # Detect smile
        if self._detect_smile(landmarks, frame.shape):
            actions.append("smile")
        
        self.blink_cooldown = max(0, self.blink_cooldown - 1)
        return actions

    def _get_landmarks(
        self,
        frame: cv2.Mat,
        face_box: Tuple[int, int, int, int],
    ) -> Optional[any]:
        """Extract facial landmarks using MediaPipe."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Use the first face (we're tracking one locked face)
        return results.multi_face_landmarks[0]

    def _detect_movement(
        self,
        landmarks: any,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[str]:
        """Detect left/right face movement."""
        height, width = frame_shape[:2]
        
        # Calculate face center using nose tip
        nose_tip = landmarks.landmark[1]
        center_x = nose_tip.x * width
        center_y = nose_tip.y * height
        
        if self.prev_center is not None:
            dx = center_x - self.prev_center[0]
            
            if abs(dx) > self.movement_threshold:
                self.prev_center = (center_x, center_y)
                if dx > 0:
                    return "moved_right"
                else:
                    return "moved_left"
        
        self.prev_center = (center_x, center_y)
        return None

    def _detect_blink(
        self,
        landmarks: any,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """Detect eye blink using eye aspect ratio."""
        if self.blink_cooldown > 0:
            return False
        
        height, width = frame_shape[:2]
        
        # Calculate left eye aspect ratio
        left_ratio = self._eye_aspect_ratio(
            landmarks, self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER, width, height
        )
        
        # Calculate right eye aspect ratio
        right_ratio = self._eye_aspect_ratio(
            landmarks, self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER, width, height
        )
        
        if left_ratio is None or right_ratio is None:
            return False
        
        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        # Detect blink: ratio drops below threshold
        if self.prev_left_eye_ratio is not None:
            if avg_ratio < self.blink_threshold and self.prev_left_eye_ratio >= self.blink_threshold:
                self.blink_cooldown = 10  # Prevent multiple detections
                self.prev_left_eye_ratio = avg_ratio
                self.prev_right_eye_ratio = avg_ratio
                return True
        
        self.prev_left_eye_ratio = left_ratio
        self.prev_right_eye_ratio = right_ratio
        return False

    def _eye_aspect_ratio(
        self,
        landmarks: any,
        upper_indices: List[int],
        lower_indices: List[int],
        width: int,
        height: int,
    ) -> Optional[float]:
        """Calculate eye aspect ratio (vertical distance / horizontal distance)."""
        try:
            # Get vertical distance
            upper_points = [landmarks.landmark[i] for i in upper_indices]
            lower_points = [landmarks.landmark[i] for i in lower_indices]
            
            vertical_dist = 0.0
            for up, low in zip(upper_points, lower_points):
                dy = abs(up.y - low.y) * height
                vertical_dist += dy
            vertical_dist /= len(upper_points)
            
            # Get horizontal distance (use eye width)
            if upper_indices == self.LEFT_EYE_UPPER:
                left_idx, right_idx = 33, 133
            else:
                left_idx, right_idx = 362, 263
            
            left_point = landmarks.landmark[left_idx]
            right_point = landmarks.landmark[right_idx]
            horizontal_dist = abs(right_point.x - left_point.x) * width
            
            if horizontal_dist == 0:
                return None
            
            return vertical_dist / horizontal_dist
        except Exception:
            return None

    def _detect_smile(
        self,
        landmarks: any,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """Detect smile using mouth aspect ratio."""
        height, width = frame_shape[:2]
        
        try:
            # Get mouth corners and top/bottom
            left_corner = landmarks.landmark[self.MOUTH_LEFT]
            right_corner = landmarks.landmark[self.MOUTH_RIGHT]
            top = landmarks.landmark[self.MOUTH_TOP]
            bottom = landmarks.landmark[self.MOUTH_BOTTOM]
            
            # Calculate mouth width and height
            mouth_width = abs(right_corner.x - left_corner.x) * width
            mouth_height = abs(bottom.y - top.y) * height
            
            if mouth_height == 0:
                return False
            
            # Smile detection: width/height ratio increases
            mouth_ratio = mouth_width / mouth_height
            
            # Simple threshold-based detection
            return mouth_ratio > self.smile_threshold / 0.1  # Adjusted for better detection
        except Exception:
            return False

    def reset(self) -> None:
        """Reset detector state."""
        self.prev_center = None
        self.prev_left_eye_ratio = None
        self.prev_right_eye_ratio = None
        self.blink_cooldown = 0

    def __del__(self) -> None:
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
