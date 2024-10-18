import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


class Visualizer:
    def __init__(self):
        self.bg_color = (0, 0, 0)  # Black background
        self.face_color = (0, 255, 255)  # Cyan for face
        self.left_hand_color = (255, 0, 255)  # Magenta for left hand
        self.right_hand_color = (0, 255, 0)  # Green for right hand

    def draw(self, image, results):
        h, w, _ = image.shape
        output_image = np.zeros((h, w, 3), dtype=np.uint8)
        output_image[:] = self.bg_color

        if 'face_landmarks' in results and results['face_landmarks']:
            self._draw_face_mesh(output_image, results['face_landmarks'])

        if 'left_hand_landmarks' in results and results['left_hand_landmarks']:
            self._draw_hand_landmarks(output_image, results['left_hand_landmarks'], self.left_hand_color)
        if 'right_hand_landmarks' in results and results['right_hand_landmarks']:
            self._draw_hand_landmarks(output_image, results['right_hand_landmarks'], self.right_hand_color)


        return output_image

    def _draw_face_mesh(self, image, landmarks):
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]

            start_x = int(start_point.x * image.shape[1])
            start_y = int(start_point.y * image.shape[0])
            end_x = int(end_point.x * image.shape[1])
            end_y = int(end_point.y * image.shape[0])

            cv2.line(image, (start_x, start_y), (end_x, end_y), self.face_color, 1)

    def _draw_hand_landmarks(self, image, landmarks, color):
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]

            start_x = int(start_point.x * image.shape[1])
            start_y = int(start_point.y * image.shape[0])
            end_x = int(end_point.x * image.shape[1])
            end_y = int(end_point.y * image.shape[0])

            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

        # Draw circles at each landmark point for better visibility
        for landmark in landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 3, color, -1)
