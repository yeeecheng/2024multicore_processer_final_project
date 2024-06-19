#ifndef CANNY_CUDA_V1_H
#define CANNY_CUDA_V1_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
using namespace cv;

#define THREAD_SIZE 32

__global__ void gray(uchar3* ,unsigned char* ,int ,int );
__global__ void gaussian_blur(unsigned char* ,unsigned char* ,int ,int ,int ,double );
__global__ void sobel_gradient(unsigned char* ,unsigned char* ,int ,int, float*);
__global__ void non_maximum_suppression(unsigned char* ,unsigned char* ,int ,int ,float* ,int );
__global__ void double_threshold(unsigned char* ,unsigned char* ,int ,int ,unsigned char* ,float ,float );

Mat canny_cuda(Mat img, int kernel_size= 5, double sigma= 1.4, int threshold= 60, int low_threshold= 50, int high_threshold= 64){
    // malloc memory

    // BGR img
    uchar3 *org_img;
    // binary img
    unsigned char *gray_img, *gaussian_img, *sobel_gradient_img, *nms_img, *double_threshold_img;
    int width = img.rows, height = img.cols;
    //printf("w: %d, h: %d\n", width, height);

    // cuda memory malloc
    cudaMalloc(&org_img, width * height * sizeof(uchar3));
    cudaMalloc(&gray_img, width * height * sizeof(unsigned char));
    cudaMalloc(&gaussian_img, width * height * sizeof(unsigned char));
    cudaMalloc(&sobel_gradient_img, width * height * sizeof(unsigned char));
    cudaMalloc(&nms_img, width * height * sizeof(unsigned char));
    cudaMalloc(&double_threshold_img, width * height * sizeof(unsigned char));

    // data copy
    cudaMemcpy(org_img, img.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice); 

    dim3 blocks(32, 32);
    dim3 grids((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1)  / blocks.y);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // gray 
    gray<<<grids, blocks>>>(org_img, gray_img, width, height);
    // gaussian blur
    gaussian_blur<<<grids, blocks , kernel_size * kernel_size * sizeof(double)>>>(gray_img, gaussian_img, width, height, kernel_size, sigma);
    // // sobel gradient
    float* theta;
    cudaMalloc(&theta, width * height * sizeof(float));
    sobel_gradient<<<grids, blocks>>>(gaussian_img, sobel_gradient_img, width, height, theta);
    // // NSM
    non_maximum_suppression<<<grids, blocks>>>(sobel_gradient_img, nms_img, width, height, theta, threshold);
    // // double threshold (final image)
    unsigned char* edge;
    cudaMalloc(&edge, width * height * sizeof(unsigned char));
    double_threshold<<<grids, blocks>>>(nms_img, double_threshold_img, width, height, edge, low_threshold, high_threshold);
    
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	// Get stop time
	
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf(" Excuetion Time on GPU: %3.20f s\n",elapsedTime/1000);

    Mat final_img(width, height, CV_8UC1);
    cudaMemcpy(final_img.data, gaussian_img, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(gray_img);
    cudaFree(gaussian_img);
    cudaFree(sobel_gradient_img);
    cudaFree(nms_img);
    cudaFree(double_threshold_img);

    return final_img;
}   


__global__ void gray(uchar3 *org_img, unsigned char* gray_img, int width, int height){

    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if(tid != 0 || blockId != 0) return;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            uchar3 rgb_pixel = org_img[i * width + j];
            gray_img[i * width + j] = (0.299f * rgb_pixel.x + 0.587f * rgb_pixel.y + 0.114f * rgb_pixel.z);
        }
    }

}

__global__ void gaussian_blur(unsigned char* gray_img, unsigned char* gaussian_blur_img, int width, int height, int kernel_size, double sigma){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if(tid != 0 || blockId != 0) return;
   
    /* get gaussian kernel */
    extern __shared__ double kernel[];
    double sum = 0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            kernel[i * kernel_size + j] =  (1.0 / (2 * M_PI * sigma * sigma)) * exp(-((i - kernel_size / 2) * (i - kernel_size / 2) + (j - kernel_size / 2) * (j - kernel_size / 2)) / (2 * sigma * sigma));
            sum += kernel[i * kernel_size + j];
        }
    }
    // normalize
    sum = 1./sum;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            kernel[i * kernel_size + j] *= sum;
        }
    }  

    /* blur */ 
    int m = kernel_size / 2;
    for(int i = m; i < height - m ; i++){
        for(int j = m; j < width - m ; j++){
            double val = 0;
            for(int ii = -m; ii < m; ii++){
                for(int jj = -m; jj < m; jj++){
                    val += gray_img[(i + ii) * width + (j + jj)] * kernel[(ii + m) * kernel_size + jj + m];
                }
            }

            gaussian_blur_img[i * width + j] = (unsigned char) val;
        }
    }

    for(int i = m; i < height - m ; i++){
        for(int j = m; j < width - m ; j++){
            if(gaussian_blur_img[i * width + j] == 0){
                printf("%d %d\n",i, j);
            }
        }
    }
   
}

__global__ void sobel_gradient(unsigned char* gaussian_blur_img, unsigned char* sobel_gradient_img, int width, int height, float* theta){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if(tid != 0 || blockId != 0) return;

    // sobel_x / sobel_y kernel
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
 
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            int gx = 0, gy = 0;
            for(int ii = -1; ii <= 1; ii++){
                for(int jj = -1; jj <= 1; jj++){
                    int val = gaussian_blur_img[(i + ii) * width + (j + jj)];
                    gx += val * sobel_x[ii + 1][jj + 1];
                    gy += val * sobel_y[ii + 1][jj + 1];
                }
            }
            sobel_gradient_img[i * width + j] = (unsigned char)sqrtf(gx * gx + gy * gy);
            theta[i * width + j] = atan2f(gy, gx) * 180 / M_PI;
        }
    }
}

__global__ void non_maximum_suppression(unsigned char* sobel_gradient_img, unsigned char* non_maximum_suppression_img, int width, int height, float* theta, int threshold){
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if(tid != 0 || blockId != 0) return;

    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            float angle = theta[i * width + j];
            double g1, g2, d_temp, weight;
            // horizon
            if((22.5 > angle && angle >= 0) || (0 >= angle && angle > -22.5) || 
            (180 > angle && angle >= 157.5) || (-157.5 >= angle && angle > -180)){
                g1 = sobel_gradient_img[i * width + j - 1];
                g2 = sobel_gradient_img[i * width + j + 1];
                weight = fabs(tanf(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (sobel_gradient_img[i * width + j] > d_temp){
                    non_maximum_suppression_img[i * width + j] = sobel_gradient_img[i * width + j];
                }
                else{
                    non_maximum_suppression_img[i * width + j] = 0;
                }
            }
            // diag_up
            else if((67.5 > angle && angle >= 22.5) || (-112.5 >= angle && angle > -157.5)){
                g1 = sobel_gradient_img[(i - 1) * width + j + 1];
                g2 = sobel_gradient_img[(i + 1) * width + j - 1];
                weight = fabs(tanf(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (sobel_gradient_img[i * width + j] > d_temp){
                    non_maximum_suppression_img[i * width + j] = sobel_gradient_img[i * width + j];
                }
                else{
                    non_maximum_suppression_img[i * width + j] = 0;
                }
            }
            // vertical
            else if((112.5 > angle && angle >= 67.5) || (-67.5 >= angle && angle > -112.5)){
                g1 = sobel_gradient_img[(i - 1) * width + j];
                g2 = sobel_gradient_img[(i + 1) * width + j];
                weight = fabs(tanf(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (sobel_gradient_img[i * width + j] > d_temp){
                    non_maximum_suppression_img[i * width + j] = sobel_gradient_img[i * width + j];
                }
                else{
                    non_maximum_suppression_img[i * width + j] = 0;
                }
            }
            // diag_down
            else if((157.5 > angle && angle >= 112.5) || (-22.5 >= angle && angle > -67.5)){
                g1 = sobel_gradient_img[(i - 1) * width + j - 1];
                g2 = sobel_gradient_img[(i + 1) * width + j + 1];
                weight = fabs(tanf(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (sobel_gradient_img[i * width + j] > d_temp){
                    non_maximum_suppression_img[i * width + j] = sobel_gradient_img[i * width + j];
                }
                else{
                    non_maximum_suppression_img[i * width + j] = 0;
                }
            }
        }
    }
}

__global__ void double_threshold(unsigned char* non_maximum_suppression_img, unsigned char* double_threshold_img, int width, int height, unsigned char* edge, float low_threshold, float high_threshold){
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if(tid != 0 || blockId != 0) return;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            // strong edge
            if(non_maximum_suppression_img[i * width + j] >= high_threshold){
                edge[i * width + j] = 255;
            }
            // weak edge
            else if(non_maximum_suppression_img[i * width + j] >= low_threshold){
                edge[i * width + j] = 128;
            }
            else{
                edge[i * width + j] = 0;
            }
        }
    }
    
    //edge tracking

    int pos[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, 
                     { 0, -1}, { 0, 0}, { 0, 1}, 
                     { 1, -1}, { 1, 0}, { 1, 1}}; 
    
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            if(edge[i * width + j] == 255){
                double_threshold_img[i * width + j] = edge[i * width + j];
                continue;
            }
            
            bool flag = false;
            for(int p = 0; p <= 8; p++){
                flag |= (edge[(i + pos[p][0] ) * width + (j + pos[p][1])] == 255);
            }
         
            int val = 0;
            if(flag) val = 255;
    
            double_threshold_img[i * width + j] = val;
        }
    }
}

#endif
