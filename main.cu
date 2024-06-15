#include <stdio.h>
#include <stdlib.h>
#include "BMP.h"

void Gray(BMP* bmp){

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
    // writing in file per pixel
    fwrite(data, 1, (bmp->info_header->img_size / 3), out);
}


void GaussianBlur(){

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
