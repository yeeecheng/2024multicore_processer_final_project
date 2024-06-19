
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>

using namespace cv;

int main(){

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Cannot open camera\n");
        return;
    }

    Mat frame;
	while(true){
        
		bool ret = cap.read(frame);
		if(!ret){
			printf("Cannot receive frame.\n");
			break;
		}
        clock_t start, end;
		resize(frame, frame, Size(1024, 1024), 0, 0, INTER_LINEAR);		
        start = clock();
        cvtColor( frame, frame, COLOR_BGR2GRAY );
        GaussianBlur(frame, frame, Size(5, 5), 1.4, 1.4);
        Canny( frame, frame, 64, 50);
        end = clock();
        printf("%.4lf\n", (double) (end - start) / CLOCKS_PER_SEC);
        imshow("live", frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }
}