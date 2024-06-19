#ifndef CANNY_CUDA_V4_H
#define CANNY_CUDA_V4_H

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
            //cudaDeviceSynchronize();
            // gaussian blur
            //printf("gaussian blur...\n");
            gaussian_blur<<<grids, blocks , (THREAD_SIZE + 4) * (THREAD_SIZE + 4) * sizeof(unsigned char)>>>(gray_img, gaussian_img, width, height, kernel_size, sigma);
            //cudaDeviceSynchronize();
            // sobel gradient
            //printf("sobel gradient...\n");
            sobel_gradient<<<grids, blocks>>>(gaussian_img, sobel_gradient_img, width, height, theta);
            //cudaDeviceSynchronize();
            // NSM
            //printf("NSM...\n");
            non_maximum_suppression<<<grids, blocks>>>(sobel_gradient_img, nms_img, width, height, theta, threshold);
            //cudaDeviceSynchronize();
            // double threshold (final image)
            //printf("double threshold...\n");
            double_threshold<<<grids, blocks>>>(nms_img, double_threshold_img, width, height, edge, low_threshold, high_threshold);
            //cudaDeviceSynchronize();
            /*
            */
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            // Get stop time
            
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf(" Excuetion Time on GPU: %3.20f s\n",elapsedTime/1000);

            Mat final_img(width, height, CV_8UC1);
            cudaMemcpy(final_img.data, double_threshold_img, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            // imwrite("../img.jpg", final_img);
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
    int private_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int global_tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x *  blockDim.x + threadIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    
    if(global_tid >= height * width){
        return;
    }

    __shared__ uchar3 shared_org_img[THREAD_SIZE * THREAD_SIZE];
    
    // confirm directly mapping
    if(gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE >= width * height){
        
        shared_org_img[private_tid] = org_img[global_tid];
        __syncthreads();

        uchar3 rgb_pixel = shared_org_img[private_tid];
        gray_img[global_tid] = (0.299f * rgb_pixel.x + 0.587f * rgb_pixel.y + 0.114f * rgb_pixel.z);
    
    }

}

__global__ void gaussian_blur(unsigned char* gray_img, unsigned char* gaussian_blur_img, int width, int height, int kernel_size, double sigma){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int private_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int global_tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x *  blockDim.x + threadIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    
    if(global_tid >= height * width){
        return;
    }

    /* gaussian kernel */
    double kernel[5][5] = { {0.012146, 0.026110, 0.033697 ,0.026110 ,0.012146},
                            {0.026110 ,0.056127, 0.072438 ,0.056127 ,0.026110},
                            {0.033697, 0.072438 ,0.093487 ,0.072438 ,0.033697}, 
                            {0.026110 ,0.056127 ,0.072438 ,0.056127, 0.026110}, 
                            {0.012146 ,0.026110 ,0.033697 ,0.026110, 0.012146}};

    int m = kernel_size / 2;
    int y = global_tid / width, x = global_tid % width;
    int inner_y = threadIdx.y, inner_x =  threadIdx.x;
    int out_y = y - m, out_x = x - m;
    int expand_width = (THREAD_SIZE + 2 * m);
    // all needed calculate unit 
    extern __shared__ unsigned char shared_gray_img[];
    

    if(gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE >= width * height){

       
        if(inner_y < m){
   
            shared_gray_img[expand_width * (inner_y) + m + inner_x] = gray_img[global_tid];
            shared_gray_img[expand_width * (inner_y + m + THREAD_SIZE) + m + inner_x] = gray_img[global_tid];
        
        }

        if(inner_x < m){
           
            shared_gray_img[expand_width * (inner_y + m) + inner_x] = gray_img[global_tid];
            shared_gray_img[expand_width * (inner_y + m) + inner_x + THREAD_SIZE + m] = gray_img[global_tid];
       
        }
        if(inner_y < m && inner_x < m){
            shared_gray_img[expand_width * inner_y + inner_x] =  gray_img[global_tid];
            shared_gray_img[expand_width * (inner_y + THREAD_SIZE + m) + inner_x] = gray_img[global_tid];
            shared_gray_img[expand_width * inner_y + inner_x + THREAD_SIZE + m] =gray_img[global_tid];
            shared_gray_img[expand_width * (inner_y + THREAD_SIZE + m) + inner_x + THREAD_SIZE + m] = gray_img[global_tid];
        }

        shared_gray_img[expand_width * (m + inner_y) + (m + inner_x)] = gray_img[global_tid];
        __syncthreads();

        double val = 0;
        if( y < m || y > (height - m) || x < m || x > (width - m)){
            return;
        }

        for(int ii = -m; ii < m; ii++){
            for(int jj = -m; jj < m; jj++){
                val += shared_gray_img[expand_width * (m + inner_y + ii) + (m + inner_x + jj)] * kernel[(ii + m)][jj + m];
            }
        }
        gaussian_blur_img[y * width + x] = (unsigned char) val;
        
    }
   
}

__global__ void sobel_gradient(unsigned char* gaussian_blur_img, unsigned char* sobel_gradient_img, int width, int height, float* theta){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int private_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int global_tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x *  blockDim.x + threadIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    
    if(global_tid >= height * width){
        return;
    }

    // sobel_x / sobel_y kernel
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    __shared__ unsigned char shared_gaussian_blur_img[THREAD_SIZE * THREAD_SIZE];


    if(gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE >= width * height){
                
        shared_gaussian_blur_img[private_tid] = gaussian_blur_img[global_tid];
        __syncthreads();

        int y = global_tid / width, x = global_tid % width;
        int inner_y = private_tid / THREAD_SIZE, inner_x =  private_tid % THREAD_SIZE;
        if( y < 1 || y > (height - 1) || x < 1 || x > (width - 1)){
            return;
        }

        int gx = 0, gy = 0;
        for(int ii = -1; ii <= 1; ii++){
            for(int jj = -1; jj <= 1; jj++){
                int val;
                if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                    val = gaussian_blur_img[(y + ii) * width + (x + jj)];
                }
                else{
                    val = shared_gaussian_blur_img[(inner_y + ii) * THREAD_SIZE + (inner_x + jj)];
                }
                gx += val * sobel_x[ii + 1][jj + 1];
                gy += val * sobel_y[ii + 1][jj + 1];
            }
        }
        
        sobel_gradient_img[global_tid] = (unsigned char)sqrtf(gx * gx + gy * gy);
        theta[global_tid] = atan2f(gy, gx) * 180 / M_PI;
       
    }

}

__global__ void non_maximum_suppression(unsigned char* sobel_gradient_img, unsigned char* non_maximum_suppression_img, int width, int height, float* theta, int threshold){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int private_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int global_tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x *  blockDim.x + threadIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    
    if(global_tid >= height * width){
        return;
    }

    __shared__ unsigned char shared_sobel_gradient_img[THREAD_SIZE * THREAD_SIZE];
    __shared__ float shared_theta[THREAD_SIZE * THREAD_SIZE];


    if(gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE >= width * height){
                
        shared_sobel_gradient_img[private_tid] = sobel_gradient_img[global_tid];
        shared_theta[private_tid] = theta[global_tid];
        __syncthreads();

        int y = global_tid / width, x = global_tid % width;
        int inner_y = private_tid / THREAD_SIZE, inner_x =  private_tid % THREAD_SIZE;
        if( y < 1 || y > (height - 1) || x < 1 || x > (width - 1)){
            return;
        }

        float angle = shared_theta[private_tid];
        double g1, g2, d_temp, weight;
        // horizon
        if((22.5 > angle && angle >= 0) || (0 >= angle && angle > -22.5) || 
        (180 > angle && angle >= 157.5) || (-157.5 >= angle && angle > -180)){
            if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                g1 = sobel_gradient_img[y * width + x - 1];
                g2 = sobel_gradient_img[y * width + x + 1];
            }
            else{
                g1 = shared_sobel_gradient_img[inner_y * THREAD_SIZE + inner_x - 1];
                g2 = shared_sobel_gradient_img[inner_y * THREAD_SIZE + inner_x + 1];
            }
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
        // diag_up
        else if((67.5 > angle && angle >= 22.5) || (-112.5 >= angle && angle > -157.5)){
            if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                g1 = sobel_gradient_img[(y - 1) * width + x + 1];
                g2 = sobel_gradient_img[(y + 1) * width + x - 1];
            }
            else{
                g1 = shared_sobel_gradient_img[(inner_y - 1) * THREAD_SIZE + inner_x + 1];
                g2 = shared_sobel_gradient_img[(inner_y + 1) * THREAD_SIZE + inner_x - 1];
            }
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;

        }
        // vertical
        else if((112.5 > angle && angle >= 67.5) || (-67.5 >= angle && angle > -112.5)){
            if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                g1 = sobel_gradient_img[(y - 1) * width + x];
                g2 = sobel_gradient_img[(y + 1) * width + x];
            }
            else {
                g1 = shared_sobel_gradient_img[(inner_y - 1) * THREAD_SIZE + inner_x];
                g2 = shared_sobel_gradient_img[(inner_y + 1) * THREAD_SIZE + inner_x];
            }
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
        // diag_down
        else if((157.5 > angle && angle >= 112.5) || (-22.5 >= angle && angle > -67.5)){
            if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                g1 = sobel_gradient_img[(y - 1) * width + x - 1];
                g2 = sobel_gradient_img[(y + 1) * width + x + 1];
            }
            else{
                g1 = shared_sobel_gradient_img[(inner_y - 1) * THREAD_SIZE + inner_x - 1];
                g2 = shared_sobel_gradient_img[(inner_y + 1) * THREAD_SIZE + inner_x + 1];
            }
            weight = fabs(tanf(angle * M_PI / 180));
            d_temp = weight * g1 + (1 - weight) * g2;
        }
    
        non_maximum_suppression_img[global_tid] = 0;
        if (shared_sobel_gradient_img[private_tid] > d_temp){
            non_maximum_suppression_img[global_tid] = shared_sobel_gradient_img[private_tid];
        }
    
    }

}

__global__ void double_threshold(unsigned char* non_maximum_suppression_img, unsigned char* double_threshold_img, int width, int height, unsigned char* edge, float low_threshold, float high_threshold){
    
    int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    int private_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int global_tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y + blockIdx.x *  blockDim.x + threadIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    
    if(global_tid >= height * width){
        return;
    }

    __shared__ unsigned char shared_non_maximum_suppression_img[THREAD_SIZE * THREAD_SIZE];
    __shared__ float shared_edge[THREAD_SIZE * THREAD_SIZE];


    if(gridDim.x * gridDim.y * THREAD_SIZE * THREAD_SIZE >= width * height){
                
        shared_non_maximum_suppression_img[private_tid] = non_maximum_suppression_img[global_tid];
        shared_edge[private_tid] = edge[global_tid];
        __syncthreads();

        // strong edge
        if(shared_non_maximum_suppression_img[private_tid] >= high_threshold){
            shared_edge[private_tid] = 255;
        }
        // weak edge
        else if(shared_non_maximum_suppression_img[private_tid] >= low_threshold){
            shared_edge[private_tid] = 128;
        }
        else{
            shared_edge[private_tid] = 0;
        }

        //edge tracking

        int pos[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, 
                        { 0, -1}, { 0, 0}, { 0, 1}, 
                        { 1, -1}, { 1, 0}, { 1, 1}}; 

        int y = global_tid / width, x = global_tid % width;
        int inner_y = private_tid / THREAD_SIZE, inner_x =  private_tid % THREAD_SIZE;
        if( y < 1 || y > (height - 1) || x < 1 || x > (width - 1)){
            return;
        }
        // if it's strong;
        if(shared_edge[private_tid] == 255){
            double_threshold_img[global_tid] = shared_edge[private_tid];
            return;
        }
        
        bool flag = false;
        for(int p = 0; p <= 8; p++){
            if(inner_y - 1 < 0 || inner_y + 1 >= THREAD_SIZE || inner_x - 1 < 0 || inner_x + 1 >= THREAD_SIZE){
                flag |= (edge[(y + pos[p][0] ) * width + (x + pos[p][1])] == 255);
            }
            else{
                flag |= (shared_edge[(inner_y + pos[p][0] ) * THREAD_SIZE + (inner_x + pos[p][1])] == 255);
            }
        }
        
        unsigned char val = 0;
        if(flag) val = 255;
      
        double_threshold_img[global_tid] = val;

    }

}

#endif
