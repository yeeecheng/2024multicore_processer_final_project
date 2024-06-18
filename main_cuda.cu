#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "include/canny_cuda_v1.h"
#include <iostream>
using namespace cv;


int main() {

    // load img
    // Mat img = imread("../img.jpg", IMREAD_COLOR);
   
    // if (img.empty()) {
    //     printf("Cannot load image");
    //     return -1;
    // }
	// //resize(img, img, Size(img.cols/10, img.rows/10), 0, 0, INTER_LINEAR);
	// img = canny_cuda(img, 5, 1.4, 60, 50, 64);
	// imshow("test", img);
	// waitKey(0);

	VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Cannot open camera\n");
        return 1;
    }

	Mat frame;

	while(true){
		bool ret = cap.read(frame);
		if(!ret){
			printf("Cannot receive frame.\n");
			break;
		}

		resize(frame,frame,Size(1024, 1024),0,0,INTER_LINEAR);		
		frame = canny_cuda(frame, 5, 1.4, 60, 50, 64);
		imshow("live", frame);

		if (waitKey(1) == 'q') {
            break;
        }
		//waitKey(0);
	}
    return 0;
}
