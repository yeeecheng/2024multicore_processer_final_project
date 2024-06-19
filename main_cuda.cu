#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// #include "include/canny_cuda_v1.h"
#include "include/canny_cuda_v2.h"
// #include "include/canny_cuda_v3.h"
// #include "include/canny_cuda_v4.h"
using namespace cv;


int main() {

	canny_cuda_streaming(1024, 1024);
    return 0;
}
