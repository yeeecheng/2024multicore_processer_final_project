#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// #include "include/canny_cuda_v1.h"
#include "include/canny_cuda_v2.h"
// #include "include/canny_cuda_v3.h"

using namespace cv;


int main() {

    // load img
    // Mat img = imread("../frame_save/frame1.bmp", IMREAD_COLOR);
   
    // if (img.empty()) {
    //     printf("Cannot load image");
    //     return -1;
    // }
	// img = canny_cuda(img, 5, 1.4, 60, 50, 64);
	// imshow("test", img);
	// waitKey(0);

	// VideoCapture cap(0);
    // if (!cap.isOpened()) {
    //     printf("Cannot open camera\n");
    //     return 1;
    // }


	canny_cuda_streaming(2048, 2048);
    return 0;
}
