#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

void getGaussianKernel(double** kernel, int size, double sigma){
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

    // calculate sobel gradient. 

    // sobel_x / sobel_y kernel
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
        }
    }

    fwrite(G_data, 1, (bmp->info_header->img_size), out);
    sobel_gradient_bmp->data = G_data;

    return sobel_gradient_bmp;
}


BMP* NonMaximumSuppression(BMP* bmp, float* theta, int threshold){

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
    
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            float angle = theta[i * width + j];
            double g1, g2, d_temp, weight;
            // horizon
            if((22.5 > angle && angle >= 0) || (0 >= angle && angle > -22.5) || 
            (180 > angle && angle >= 157.5) || (-157.5 >= angle && angle > -180)){
                g1 = bmp->data[i * width + j - 1];
                g2 = bmp->data[i * width + j + 1];
                weight = fabs(tan(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (bmp->data[i * width + j] > d_temp){
                    data[i * width + j] = bmp->data[i * width + j];
                }
                else{
                    data[i * width + j] = 0;
                }
            }
            // diag_up
            else if((67.5 > angle && angle >= 22.5) || (-112.5 >= angle && angle > -157.5)){
                g1 = bmp->data[(i - 1) * width + j + 1];
                g2 = bmp->data[(i + 1) * width + j - 1];
                weight = fabs(tan(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (bmp->data[i * width + j] > d_temp){
                    data[i * width + j] = bmp->data[i * width + j];
                }
                else{
                    data[i * width + j] = 0;
                }
            }
            // vertical
            else if((112.5 > angle && angle >= 67.5) || (-67.5 >= angle && angle > -112.5)){
                g1 = bmp->data[(i - 1) * width + j];
                g2 = bmp->data[(i + 1) * width + j];
                weight = fabs(tan(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (bmp->data[i * width + j] > d_temp){
                    data[i * width + j] = bmp->data[i * width + j];
                }
                else{
                    data[i * width + j] = 0;
                }
            }
            // diag_down
            else if((157.5 > angle && angle >= 112.5) || (-22.5 >= angle && angle > -67.5)){
                g1 = bmp->data[(i - 1) * width + j - 1];
                g2 = bmp->data[(i + 1) * width + j + 1];
                weight = fabs(tan(angle * M_PI / 180));
                d_temp = weight * g1 + (1 - weight) * g2;
                if (bmp->data[i * width + j] > d_temp){
                    data[i * width + j] = bmp->data[i * width + j];
                }
                else{
                    data[i * width + j] = 0;
                }
            }
        }
    }
    
    fwrite(data, 1, (bmp->info_header->img_size), out);
    non_maximum_suppression_bmp->data = data;

    return non_maximum_suppression_bmp;
}

void DoubleThresholding(BMP* bmp, float low_threshold, float high_threshold){

    BMP* double_thresholding_bmp = (BMP*)malloc(sizeof(BMP));
    double_thresholding_bmp->header = bmp->header;
    double_thresholding_bmp->info_header = bmp->info_header;
    double_thresholding_bmp->rgb_quad = bmp->rgb_quad;

    FILE* out = fopen("./doubleThresholding.bmp", "wb");
    fwrite(double_thresholding_bmp->header, sizeof(HEADER), 1, out);
    fwrite(double_thresholding_bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(double_thresholding_bmp->rgb_quad, sizeof(RGBQUAD), 256, out);

    int width = bmp->info_header->width, height = bmp->info_header->height; 
    unsigned char* data = (unsigned char*)malloc(bmp->info_header->img_size * sizeof(unsigned char));
    
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            
            // strong edge
            if(bmp->data[i * width + j] >= high_threshold){
                data[i * width + j] = 255;
            }
            // weak edge
            else if(bmp->data[i * width + j] >= low_threshold){
                data[i * width + j] = 128;
            }
            else{
                data[i * width + j] = 0;
            }

        }
    }
    

    //edge tracking

    unsigned char* edge_data = (unsigned char*)malloc(bmp->info_header->img_size * sizeof(unsigned char));

    int pos[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, 
                     { 0, -1}, { 0, 0}, { 0, 1}, 
                     { 1, -1}, { 1, 0}, { 1, 1}}; 
    
    for(int i = 1; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            
            if(data[i * width + j] == 255){
                edge_data[i * width + j] = data[i * width + j];
                continue;
            }
            
            bool flag = false;
            for(int p = 0; p <= 8; p++){
                flag |= (data[(i + pos[p][0] ) * width + (j + pos[p][1])] == 255);
            }
         
            int val = 0;
            if(flag) val = 255;
    
            edge_data[i * width + j] = val;
        }
    }

    fwrite(edge_data, 1, (bmp->info_header->img_size), out);
    double_thresholding_bmp->data = edge_data;
}

int main() {
    
    // Canny Algorithm
    BMP* bmp = OpenImg("./test.bmp");
    clock_t start, end;

    start = clock();
    BMP* gray_bmp = Gray(bmp);
    BMP* gaussian_blur_bmp = GaussianBlur(gray_bmp, 5, 1.4);

    float* theta = (float*)malloc(gaussian_blur_bmp->info_header->img_size * sizeof(float));
    BMP* sobel_gradient_bmp = SobelGradient(gaussian_blur_bmp, theta);
    BMP* non_maximum_suppression_bmp = NonMaximumSuppression(sobel_gradient_bmp, theta, 60);
    DoubleThresholding(non_maximum_suppression_bmp, 50, 64);
    end = clock();
    printf("%.4lf\n", (double) (end - start) / CLOCKS_PER_SEC);
    free(bmp);

    

    return 0;
}
