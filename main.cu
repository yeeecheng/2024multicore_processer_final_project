#include <stdio.h>
#include <stdlib.h>
#include "BMP.h"

void Gray(BMP* bmp){

    bmp->info_header->bits = 8;
    bmp->header->offset = sizeof(HEADER) + sizeof(INFO_HEADER) + 256 * sizeof(RGBQUAD);
    bmp->header->size = bmp->header->offset + bmp->info_header->img_size;
    // bits=24, 32 no need 調色盤, then gray scale need it which is R=G=B
    RGBQUAD* rgb_quad = (RGBQUAD*)malloc(256 * sizeof(RGBQUAD));
    for(int i = 0; i < 256; i++){
        rgb_quad[i].rgbBlue = rgb_quad[i].rgbGreen = rgb_quad[i].rgbRed = i;
    }

    // write into new file.
    FILE* out = fopen("./gray.bmp", "wb");    
    fwrite(bmp->header, sizeof(HEADER), 1, out);
    fwrite(bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fwrite(rgb_quad,  sizeof(RGBQUAD), 256, out);
    
    unsigned char* data = bmp->data;
    for(int i = 0; i <  bmp->info_header->img_size; i += 3){
        // 0.299 * R + 0.587 * G + 0.114 * B;
        unsigned char Y = (int)(0.299 * (float)data[i + 2] + 0.587 * (float)data[i + 1] + 0.114 * (float)data[i]);
        // writing in file per pixel
        fwrite(&Y, 1, 1, out);
    
    }
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
    
    BMP* bmp = OpenImg("./frame_out/frame1.bmp");
    PrintImgInfo(bmp);
    Gray(bmp);


    free(bmp);
    

    return 0;
}
