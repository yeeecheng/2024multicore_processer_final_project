#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceId = 0; // GPU device ID, 0 is typically the first GPU
    cudaDeviceProp deviceProp;

    cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "Device Name: " << deviceProp.name << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimensions (x, y, z): (" 
              << deviceProp.maxThreadsDim[0] << ", " 
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid size (x, y, z): (" 
              << deviceProp.maxGridSize[0] << ", " 
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << ")" << std::endl;

    return 0;
}