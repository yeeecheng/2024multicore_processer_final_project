#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera\n";
        return 1;
    }

    Mat frame;
    Mat gray;
    //namedWindow("live", WINDOW_AUTOSIZE); // 命名一個視窗，可不寫
    while (true) {
        // 擷取影像
        bool ret = cap.read(frame); // or cap >> frame;
        if (!ret) {
            cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }
        
        // 彩色轉灰階
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 顯示圖片
        imshow("live", frame);
        //imshow("live", gray);

        // 按下 q 鍵離開迴圈
        if (waitKey(1) == 'q') {
            break;
        }
    }
    // VideoCapture 會自動在解構子裡釋放資源
    return 0;
}