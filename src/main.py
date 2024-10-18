import cv2
import time
from tracker import FaceHandTracker
from visualizer import Visualizer
import platform

def get_camera():
    if platform.system() == 'Darwin':  # macOS
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                return cap
        raise RuntimeError("Unable to find an available camera")
    else:
        return cv2.VideoCapture(0)

def main():
    cap = get_camera()
    
    tracker = FaceHandTracker()
    visualizer = Visualizer()

    cv2.namedWindow('Face and Hand Tracker')

    print("- Press 'q' to exit -")
    while cap.isOpened():
        start_time = time.time()

        success, image = cap.read()
        if not success:
            print("Processing")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = tracker.process(image)

        output_image = visualizer.draw(image, results)

        cv2.imshow('Face and Hand Tracker', output_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # Calculate the time taken for processing this frame
        process_time = time.time() - start_time

        # If processing was faster than the frame time, wait for the remainder
        if process_time < 1 / 60:
            time.sleep(1 / 60 - process_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
