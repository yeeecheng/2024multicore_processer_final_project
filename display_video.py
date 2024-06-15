import cv2
import time
import numpy as np
import threading

def StartVideo():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #start_time = time.time()
    #duration  = 0.5
    frame_num = 1
    while frame_num <= 10000:
        ret, frame = cap.read()             
        if not ret:
            print("Cannot receive frame")   
            break
        #cv2.imshow('video', frame)
        frame = cv2.resize(frame, (388, 388))
        cv2.imwrite(f"./frame_out/frame{frame_num}.bmp", frame)     
        if cv2.waitKey(1) == ord('q'):      
            break
    
    
        frame_num += 1
    cap.release()                           
    cv2.destroyAllWindows()                 


def DisplayVideo():
    
    for frame_num in range(1, 10001):
        frame = cv2.imread(f"./frame_out/frame{frame_num}.bmp")
        #cv2.imshow("show", frame)
        
        # cv2.waitKey(0)
    cv2.destroyWindow('show')
if __name__ == "__main__":
    
    #start_t = threading.Thread(target= StartVideo)
    show_t = threading.Thread(target= DisplayVideo)
    #start_t.start()
    time.sleep(3)    
    show_t.start()