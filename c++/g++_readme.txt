g++ -o test_cuda  test_cuda.cpp -I../libtorch/include/torch/csrc/api/include/ -I../libtorch/include/ -L../libtorch/lib -ltorch  -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc


g++ -o main  main.cpp -I./libtorch/include/torch/csrc/api/include/ -I./libtorch/include/ -L./libtorch/lib -ltorch  -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc



export LD_LIBRARY_PATH=./libtorch/lib