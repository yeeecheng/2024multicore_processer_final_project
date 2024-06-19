#ifndef CANNY_CUDA_V2_H
#define CANNY_CUDA_V2_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
using namespace cv;

#define THREAD_SIZE 16

__global__ void gray(uchar3* ,unsigned char* ,int ,int );
__global__ void gaussian_blur(unsigned char* ,unsigned char* ,int ,int ,int ,double );
__global__ void sobel_gradient(unsigned char* ,unsigned char* ,int ,int, float*);
__global__ void non_maximum_suppression(unsigned char* ,unsigned char* ,int ,int ,float* ,int );
__global__ void double_threshold(unsigned char* ,unsigned char* ,int ,int ,unsigned char* ,float ,float );

void canny_cuda_streaming(int width, int height, int kernel_size= 5, double sigma= 1.4, int threshold= 60, int low_threshold= 50, int high_threshold= 64){
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Cannot open camera\n");
        return;
    }

    // BGR img
    uchar3 *org_img;
    // binary img
    unsigned char *gray_img, *gaussian_img, *sobel_gradient_img, *nms_img, *edge, *double_threshold_img;
    float* theta;

    // cuda memory malloc
    cudaError_t R ;
    R = cudaMalloc(&org_img, width * height * sizeof(uchar3));
    printf(" org_img : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&gray_img, width * height * sizeof(unsigned char));
    printf(" gray_img : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&gaussian_img, width * height * sizeof(unsigned char));
    printf(" gaussian_img : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&theta, width * height * sizeof(float));
    printf(" theta : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&sobel_gradient_img, width * height * sizeof(unsigned char));
    printf(" sobel_gradient_img : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&nms_img, width * height * sizeof(unsigned char));
    printf(" nms_img : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&edge, width * height * sizeof(unsigned char));
    printf(" edge : %s\n",cudaGetErrorString(R));
    R = cudaMalloc(&double_threshold_img, width * height * sizeof(unsigned char));
    printf(" double_threshold_img : %s\n",cudaGetErrorString(R));


	Mat frame;
	while(true){

		bool ret = cap.read(frame);
		if(!ret){
			printf("Cannot receive frame.\n");
			break;
		}
		resize(frame, frame, Size(width, height), 0, 0, INTER_LINEAR);		
	
        // data copy
        R = cudaMemcpy(org_img, frame.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice); 
        
        if (R != cudaSuccess) {
            printf(" org_img : %s\n",cudaGetErrorString(R));
        }
        else{

        
            dim3 blocks(THREAD_SIZE, THREAD_SIZE);
            dim3 grids((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1)  / blocks.y);
            
            cudaEvent_t start,stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            // gray 
            //printf("gray...\n");
            gray<<<grids, blocks>>>(org_img, gray_img, width, height);
            // cudaDeviceSynchronize();
            // gaussian blur
            //printf("gaussian blur...\n");
            gaussian_blur<<<grids, blocks>>>(gray_img, gaussian_img, width, height, kernel_size, sigma);
            // cudaDeviceSynchronize();
            // sobel gradient
            //printf("sobel gradient...\n");
            sobel_gradient<<<grids, blocks>>>(gaussian_img, sobel_gradient_img, width, height, theta);
            // cudaDeviceSynchronize();
            // NSM
            //printf("NSM...\n");
            non_maximum_suppression<<<grids, blocks>>>(sobel_gradient_img, nms_img, width, height, theta, threshold);
            // cudaDeviceSynchronize();
            // double threshold (final image)
            //printf("double threshold...\n");
            double_threshold<<<grids, blocks>>>(nms_img, double_threshold_img, width, height, edge, low_threshold, high_threshold);
            // cudaDeviceSynchronize();

            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            // Get stop time
            
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf(" Excuetion Time on GPU: %3.20f s\n",elapsedTime/1000);

            Mat final_img(width, height, CV_8UC1);
            cudaMemcpy(final_img.data, double_threshold_img, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            imshow("live", final_img);

            if (waitKey(1) == 'q') {
                break;
            }
        }
        
	}

    cudaFree(gray_img);
    cudaFree(gaussian_img);
    cudaFree(theta);
    cudaFree(sobel_gradient_img);
    cudaFree(nms_img);
    cudaFree(double_threshold_img);
    cudaFree(edge);
}


__global__ void gray(uchar3 *org_img, unsigned char* gray_img, int width, int height){

    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockId * (THREAD_SIZE* THREAD_SIZE) + blockDim.x * threadIdx.y + threadIdx.x;

    if(tid >= height * width){
        return;
    }

    for(int i = tid; i < height * width; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
        uchar3 rgb_pixel = org_img[i];
        if(i < width * height){
            gray_img[i] = (0.299f * rgb_pixel.x + 0.587f * rgb_pixel.y + 0.114f * rgb_pixel.z);
        }
    }

}

__global__ void gaussian_blur(unsigned char* gray_img, unsigned char* gaussian_blur_img, int width, int height, int kernel_size, double sigma){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockId * (THREAD_SIZE* THREAD_SIZE) + blockDim.x * threadIdx.y + threadIdx.x;
   
    if(tid >= height * width){
        return;
    }
    /* gaussian kernel */
    double kernel[5][5] = { {0.012146, 0.026110, 0.033697 ,0.026110 ,0.012146},
                            {0.026110 ,0.056127, 0.072438 ,0.056127 ,0.026110},
                            {0.033697, 0.072438 ,0.093487 ,0.072438 ,0.033697}, 
                            {0.026110 ,0.056127 ,0.072438 ,0.056127, 0.026110}, 
                            {0.012146 ,0.026110 ,0.033697 ,0.026110, 0.012146}};

    // /* blur */ 
    int m = kernel_size / 2;
    for(int i = tid; i < width * height; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
        int y = i / width, x = i % width;
        
        double val = 0;
        if( y < m || y > (height - m) || x < m || x > (width - m)){
            continue;
        }
        for(int ii = -m; ii < m; ii++){
            for(int jj = -m; jj < m; jj++){
                val += gray_img[(y + ii) * width + (x + jj)] * kernel[(ii + m)][jj + m];
            }
        }
        if(i < width * height){
            gaussian_blur_img[i] = (unsigned char) val;
        }
    }
   
}

__global__ void sobel_gradient(unsigned char* gaussian_blur_img, unsigned char* sobel_gradient_img, int width, int height, float* theta){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockId * (THREAD_SIZE* THREAD_SIZE) + blockDim.x * threadIdx.y + threadIdx.x;

    if(tid >= height * width){
        return;
    }

    // sobel_x / sobel_y kernel
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for(int i = tid; i < width * height; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
        int y = i / width, x = i % width;
        
        if( y < 1 || y > (height - 1) || x < 1|| x > (width - 1)){
            continue;
        }
        int gx = 0, gy = 0;
        for(int ii = -1; ii <= 1; ii++){
            for(int jj = -1; jj <= 1; jj++){
                int val = gaussian_blur_img[(y + ii) * width + (x + jj)];
                gx += val * sobel_x[ii + 1][jj + 1];
                gy += val * sobel_y[ii + 1][jj + 1];
            }
        }
        if(i < width * height){
            sobel_gradient_img[i] = (unsigned char)sqrtf(gx * gx + gy * gy);
            theta[i] = atan2f(gy, gx) * 180 / M_PI;
        }

    }

}

__global__ void non_maximum_suppression(unsigned char* sobel_gradient_img, unsigned char* non_maximum_suppression_img, int width, int height, float* theta, int threshold){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockId * (THREAD_SIZE* THREAD_SIZE) + blockDim.x * threadIdx.y + threadIdx.x;

    if(tid >= height * width){
        return;
    }

    for(int i = tid; i < width * height; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
        int y = i / width, x = i % width;
        
        if( y < 1 || y > (height - 1) || x < 1|| x > (width - 1)){
            continue;
        }
        float angle = theta[i];
        double g1, g2, d_temp, weight;
        // horizon
        if((22.5 > angle && angle >= 0) || (0 >= angle && angle > -22.5) || 
        (180 > angle && angle >= 157.5) || (-157.5 >= angle && angle > -180)){
            g1 = sobel_gradient_img[y * width + x - 1];
            g2 = sobel_gradient_img[y * width + x + 1];
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
        // diag_up
        else if((67.5 > angle && angle >= 22.5) || (-112.5 >= angle && angle > -157.5)){
            g1 = sobel_gradient_img[(y - 1) * width + x + 1];
            g2 = sobel_gradient_img[(y + 1) * width + x - 1];
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;

        }
        // vertical
        else if((112.5 > angle && angle >= 67.5) || (-67.5 >= angle && angle > -112.5)){
            g1 = sobel_gradient_img[(y - 1) * width + x];
            g2 = sobel_gradient_img[(y + 1) * width + x];
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
        // diag_down
        else if((157.5 > angle && angle >= 112.5) || (-22.5 >= angle && angle > -67.5)){
            g1 = sobel_gradient_img[(y - 1) * width + x - 1];
            g2 = sobel_gradient_img[(y + 1) * width + x + 1];
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
        if(i < width * height){
            non_maximum_suppression_img[i] = 0;
            if (sobel_gradient_img[i] > d_temp){
                non_maximum_suppression_img[i] = sobel_gradient_img[i];
            }
        }
    }
}

__global__ void double_threshold(unsigned char* non_maximum_suppression_img, unsigned char* double_threshold_img, int width, int height, unsigned char* edge, float low_threshold, float high_threshold){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockId * (THREAD_SIZE* THREAD_SIZE) + blockDim.x * threadIdx.y + threadIdx.x;

    if(tid >= height * width){
        return;
    }

    for(int i = tid; i < width * height; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
       
        // strong edge
        if(non_maximum_suppression_img[i] >= high_threshold){
            edge[i] = 255;
        }
        // weak edge
        else if(non_maximum_suppression_img[i] >= low_threshold){
            edge[i] = 128;
        }
        else{
            edge[i] = 0;
        }
    }
    
    //edge tracking

    int pos[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, 
                     { 0, -1}, { 0, 0}, { 0, 1}, 
                     { 1, -1}, { 1, 0}, { 1, 1}}; 
    
    for(int i = tid; i < width * height; i += gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE){
        int y = i / width, x = i % width;
        
        if( y < 1 || y > (height - 1) || x < 1|| x > (width - 1)){
            continue;
        }

        if(edge[i] == 255){
            double_threshold_img[i] = edge[i];
            continue;
        }
        
        bool flag = false;
        for(int p = 0; p <= 8; p++){
            flag |= (edge[(y + pos[p][0] ) * width + (x + pos[p][1])] == 255);
        }
        
        int val = 0;
        if(flag) val = 255;
        if(i < width * height){
            double_threshold_img[i] = val;
        }
    }
}

#endif
