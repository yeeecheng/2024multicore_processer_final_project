#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "BMP.h"

BMP* Gray(BMP* bmp){

    BMP* gray_bmp = (BMP*)malloc(sizeof(BMP));
    gray_bmp->header = bmp->header;
    gray_bmp->info_header = bmp->info_header;
    gray_bmp->info_header->bits = 8;
    gray_bmp->header->offset = sizeof(HEADER) + sizeof(INFO_HEADER) + 256 * sizeof(RGBQUAD);
    gray_bmp->header->size = gray_bmp->header->offset + gray_bmp->info_header->img_size;
    // bits=24, 32 no need 調色盤, then gray scale need it which is R=G=B
    RGBQUAD* rgb_quad = (RGBQUAD*)malloc(256 * sizeof(RGBQUAD));
    for(int i = 0; i < 256; i++){
        rgb_quad[i].rgbBlue = rgb_quad[i].rgbGreen = rgb_quad[i].rgbRed = i;
    }
    gray_bmp->rgb_quad = rgb_quad;
    // write into new file.
    FILE* out = fopen("./gray.bmp", "wb");    
    fwrite(gray_bmp->header, sizeof(HEADER), 1, out);
    fwrite(gray_bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(gray_bmp->rgb_quad,  sizeof(RGBQUAD), 256, out);
    
    unsigned char* data = (unsigned char*)malloc((bmp->info_header->img_size / 3) * sizeof(unsigned char));
    for(int i = 0; i <  bmp->info_header->img_size; i += 3){
        // 0.299 * R + 0.587 * G + 0.114 * B;
        data[i / 3] = (int)(0.299 * (float)bmp->data[i + 2] + 0.587 * (float)bmp->data[i + 1] + 0.114 * (float)bmp->data[i]);
    }
    // writing in file 
    fwrite(data, 1, (bmp->info_header->img_size / 3), out);
    gray_bmp->data = data;

    return gray_bmp;
}

void getGaussianKernel(double** kernel, int size, int sigma){
    if(size <= 0 || sigma == 0) return;

    double sum = 0;
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            kernel[i][j] = (1.0 / (2 * M_PI * sigma * sigma)) * exp(-((i - size / 2) * (i - size / 2) + (j - size / 2) * (j - size / 2)) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    // normalize
    sum = 1./sum;
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            kernel[i][j] *= sum;
        }
    }    
}

BMP* GaussianBlur(BMP* bmp, int size, double sigma){
    
    BMP* gaussian_blur_bmp = (BMP*)malloc(sizeof(BMP));
    gaussian_blur_bmp->header = bmp->header;
    gaussian_blur_bmp->info_header = bmp->info_header;
    gaussian_blur_bmp->rgb_quad = bmp->rgb_quad;

    FILE* out = fopen("./gaussianBlur.bmp", "wb");
    fwrite(gaussian_blur_bmp->header, sizeof(HEADER), 1, out);
    fwrite(gaussian_blur_bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(gaussian_blur_bmp->rgb_quad, sizeof(RGBQUAD), 256, out);
    sigma = sigma > 0 ? sigma : ((size - 1) * 0.5 - 1) * 0.3 + 0.8;
    
    // create gaussian kernel
    double** kernel = (double**)malloc(size * sizeof(double*));
    for(int i = 0; i < size; i++){
        kernel[i] = (double*)malloc(size * sizeof(double));
    }

    getGaussianKernel(kernel, size, sigma);

    // raw data mul kernel weight.
    int width = bmp->info_header->width, height = bmp->info_header->height;
    unsigned char* data = (unsigned char*)malloc(bmp->info_header->img_size * sizeof(unsigned char));
    data = bmp->data;

    int m = size / 2;
    for(int i = m; i < height - m; i++){
        for(int j = m; j < width - m; j++){
            
            double val = 0;
            for(int ii = -m; ii < m; ii++){
                for(int jj = -m; jj < m; jj++){
                    val += bmp->data[(i + ii) * width + (j + jj)] * kernel[ii + m][jj + m];
                }
            }
            data[i * width + j] = (unsigned char) val;
        }
    }
    
    // writing in file per pixel
    fwrite(data, 1, (bmp->info_header->img_size), out);
    gaussian_blur_bmp->data = data;
    // free
    for(int i = 0; i < size; i++){
        free(kernel[i]);
    }
    free(kernel);

    return gaussian_blur_bmp;
}

BMP* SobelGradient(BMP* bmp, float* theta){
    
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    BMP* sobel_gradient_bmp = (BMP*)malloc(sizeof(BMP));
    sobel_gradient_bmp->header = bmp->header;
    sobel_gradient_bmp->info_header = bmp->info_header;
    sobel_gradient_bmp->rgb_quad = bmp->rgb_quad;

    FILE* out = fopen("./sobelGradient.bmp", "wb");
    fwrite(sobel_gradient_bmp->header, sizeof(HEADER), 1, out);
    fwrite(sobel_gradient_bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(sobel_gradient_bmp->rgb_quad, sizeof(RGBQUAD), 256, out);

    int width = bmp->info_header->width, height = bmp->info_header->height;
    
    unsigned char* G_data = (unsigned char*)malloc(bmp->info_header->img_size * sizeof(unsigned char)); 
    // G_data = bmp->data;
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            int gx = 0, gy = 0;
            for(int ii = -1; ii <= 1; ii++){
                for(int jj = -1; jj <= 1; jj++){
                    int val = bmp->data[(i + ii) * width + (j + jj)];
                    gx += val * sobel_x[ii + 1][jj + 1];
                    gy += val * sobel_y[ii + 1][jj + 1];
                }
            }
            G_data[i * width + j] = (unsigned char)sqrt(gx * gx + gy * gy);
            theta[i * width + j] = atan2(gy, gx) * 180 / M_PI;
          
            // if(G_data[i * width + j] > 0){
            //     G_data[i * width + j] = 255;
            // }
            // printf("%d\n", G_data[i * width + j]);
        }
    }
    fwrite(G_data, 1, (bmp->info_header->img_size), out);
    sobel_gradient_bmp->data = G_data;

    return sobel_gradient_bmp;
}


void NonMaximumSuppression(BMP* bmp, float* theta, int threshold){

    BMP* non_maximum_suppression_bmp = (BMP*)malloc(sizeof(BMP));
    non_maximum_suppression_bmp->header = bmp->header;
    non_maximum_suppression_bmp->info_header = bmp->info_header;
    non_maximum_suppression_bmp->rgb_quad = bmp->rgb_quad;

    FILE* out = fopen("./nonMaximumSuppression.bmp", "wb");
    fwrite(non_maximum_suppression_bmp->header, sizeof(HEADER), 1, out);
    fwrite(non_maximum_suppression_bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(non_maximum_suppression_bmp->rgb_quad, sizeof(RGBQUAD), 256, out);

    int width = bmp->info_header->width, height = bmp->info_header->height;

    unsigned char* data = (unsigned char*)malloc(bmp->info_header->img_size * sizeof(unsigned char)); 
    // data = bmp->data;
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            float angle = theta[i * width + j];
            
            if(angle < 0){
                angle += 180;
            }

            // Find the max gradient in gradient direction（0, 45, 90, 135). if it isn't maximum gradient setting to 0.
            if(((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) ||
            ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5)) ||
            ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5)) ||
            ((angle >= 112.5 && angle < 157.5) || (angle >= 292.5 && angle < 337.5))){

                if(bmp->data[i * width + j] > bmp->data[i * width + j - 1] && bmp->data[i * width + j] > bmp->data[i * width + j + 1]){
                    data[i * width + j] = (bmp->data[i * width + j] >= threshold) ? 255 : 0;
                }
                else {
                    data[i * width + j] = 0;
                }
            }
            else{
                if(bmp->data[i * width + j] > bmp->data[(i - 1) * width + j] && bmp->data[i * width + j] > bmp->data[(i + 1)  * width + j]){
                    data[i * width + j] = (bmp->data[i * width + j] >= threshold) ? 255 : 0;
                }
                else {
                    data[i * width + j] = 0;
                }
            }
        }
    }

    for(int i = 0; i < height; i++){
        data[i * width] = 0;
        data[i * width + width - 1] = 0;
    }

    for(int j = 0; j < width; j++){
        data[j] = 0;
        data[(height - 1) * width + j] = 0;
    }
    fwrite(data, 1, (bmp->info_header->img_size), out);
    non_maximum_suppression_bmp->data = data;
}


int main() {
    
    // char get_file_path[50], out_file_path[50];
    // const char* get_root = "./frame_save/frame", *out_root = "./frame_out/frame";
    // const char* file = ".bmp";
    // for(int i = 1; i <= 1000; i++){
    //     sprintf(get_file_path, "%s%d%s", get_root, i, file);
    //     sprintf(out_file_path, "%s%d%s", out_root, i, file);
    //     printf("%s %s\n", get_file_path, out_file_path);
    //     BMP* bmp = OpenImg(get_file_path);
    //     //PrintImgInfo(bmp);
    //     SaveImg(out_file_path, bmp);
    //     free(bmp);
        
    // }
    
    BMP* bmp = OpenImg("./frame_save/frame1.bmp");
    BMP* gray_bmp = Gray(bmp);
    BMP* gaussian_blur_bmp = GaussianBlur(gray_bmp, 5, 1);
    PrintImgInfo(gaussian_blur_bmp);

    float* theta = (float*)malloc(gaussian_blur_bmp->info_header->img_size * sizeof(float));
    BMP* sobel_gradient_bmp = SobelGradient(gaussian_blur_bmp, theta);
    PrintImgInfo(sobel_gradient_bmp);
    NonMaximumSuppression(sobel_gradient_bmp, theta, 100);
    free(bmp);
    

    return 0;
}
