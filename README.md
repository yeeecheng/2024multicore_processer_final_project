## final_project

* setup environment

### step1
```=cmd
sudo apt install libopencv-dev
pkg-config --modversion opencv4
```
### step2

#### In vscode, you need to install extension modules, which are Cmake and Cmake tool, then add CMakeLists.txt in your folder.

#### Change <folder_name> and <file_name>
```=txt
cmake_minimum_required(VERSION 3.0.0)
project(<folder_name> VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE(<folder_name> <file_name>)

target_link_libraries( <folder_name> ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
```
### Step3 
#### >Cmake , then it will generate the folder named build.

### Step4
```=cmd
cd build
make
./<folder_name>
```

* setup environment
```=cmd
nvcc main.cu -o main
./main
```