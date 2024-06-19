import cv2
import time

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.resize(frame, (1024, 1024))
        cv2.imwrite("./test.bmp", frame)
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 1.4)
        edges = cv2.Canny(blur_gray, 50, 64)
     
        end_time = time.time()
        frame_time = end_time - start_time
        print(frame_time)
        #cv2.putText(edges, f'Time: {frame_time:.3f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #cv2.imshow('Original', frame)
        cv2.imshow('Canny Edges', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()