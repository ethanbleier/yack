import mediapipe as mp
import cv2

class FaceHandTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        results = {
            'face_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None
        }

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face mesh
        face_results = self.face_mesh.process(image_rgb)
        if face_results.multi_face_landmarks:
            results['face_landmarks'] = face_results.multi_face_landmarks[0]

        # Process hands
        hand_results = self.hands.process(image_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    results['left_hand_landmarks'] = hand_landmarks
                else:
                    results['right_hand_landmarks'] = hand_landmarks

        return results

    def __del__(self):
        self.face_mesh.close()
        self.hands.close()
