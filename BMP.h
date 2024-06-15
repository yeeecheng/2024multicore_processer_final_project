#ifndef BMP_H
#define BMP_H

#include <stdio.h>
#include <stdlib.h>
 

#pragma pack(push, 1)


typedef struct {
    unsigned short type;  // Magic identifier
    unsigned int size;  // File size in bytes
    unsigned short reserved1;  // Not used
    unsigned short reserved2;  // Not used
    unsigned int offset;  // Offset to image data in bytes
} HEADER;

typedef struct {
    unsigned int size;  // Header size in bytes
    int width, height;  // Width and height of image
    unsigned short planes;  // Number of colour planes
    unsigned short bits;  // Bits per pixel
    unsigned int compression;  // Compression type
    unsigned int img_size;  // Image size in bytes
    int xresolution, yresolution;  // Pixels per meter
    unsigned int ncolours;  // Number of colours
    unsigned int importantcolours;  // Important colours
} INFO_HEADER;

#pragma pack(pop)

#pragma pack(1)
typedef struct {

    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
    unsigned char rgbReserved;
}RGBQUAD;

typedef struct {

    HEADER* header;
    INFO_HEADER* info_header;
    RGBQUAD* rgb_quad;
    unsigned char* data;

} BMP;


BMP* OpenImg(char* img_path){

    FILE *file = fopen(img_path, "rb");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(1);
    }
    
    HEADER* header = (HEADER*)malloc(sizeof(HEADER));
    fread(header, sizeof(HEADER), 1, file);

    INFO_HEADER* info_header = (INFO_HEADER*)malloc(sizeof(INFO_HEADER));
    fread(info_header, sizeof(INFO_HEADER), 1, file);


    unsigned int img_size = info_header->img_size;
  
    // avoid img_size is 0
    if(img_size == 0){
        img_size = info_header->width * info_header->height * 3;
        info_header->img_size = img_size;
    }
    unsigned char* data = (unsigned char*) malloc(img_size);
    fseek(file, header->offset, SEEK_SET);
    fread(data, 1, img_size, file);

    fclose(file);

    BMP* bmp = (BMP*)malloc(sizeof(BMP));
    bmp->header = header;
    bmp->info_header = info_header;
    bmp->data = data;
    
    return bmp;
}

void SaveImg(char* img_path,  BMP* bmp){

    FILE *out = fopen(img_path, "wb");
    fwrite(bmp->header, sizeof(HEADER), 1, out);
    fwrite(bmp->info_header, sizeof(INFO_HEADER), 1, out);
    fseek(out, bmp->header->offset, SEEK_SET);

    int img_size = bmp->info_header->img_size;

    // avoid img_size is 0
    if(img_size == 0){
        img_size = bmp->info_header->width * bmp->info_header->height * 3;
    }

    fwrite(bmp->data, 1, img_size, out);
    fclose(out);
    return;
}


void PrintImgInfo(BMP* bmp){
    printf("Header: \n");
    printf("type: %u\n", bmp->header->type);
    printf("size: %u\n", bmp->header->size);
    printf("reserved1: %u\n", bmp->header->reserved1);
    printf("reserved2: %u\n", bmp->header->reserved2);
    printf("offset: %u\n", bmp->header->offset);

    printf("\nInfo Header: \n");
    printf("size: %u\n", bmp->info_header->size);
    printf("width: %d, height: %d\n", bmp->info_header->width, bmp->info_header->height);
    printf("planes: %u\n", bmp->info_header->planes);
    printf("bits: %u\n", bmp->info_header->bits);
    printf("compression: %u\n", bmp->info_header->compression);
    printf("img_size: %u\n", bmp->info_header->img_size);
    printf("xresolution: %d, yresolution: %d\n", bmp->info_header->xresolution, bmp->info_header->yresolution);
    printf("ncolours: %u\n", bmp->info_header->ncolours);
    printf("importantcolours: %u\n", bmp->info_header->importantcolours);
}


#endif