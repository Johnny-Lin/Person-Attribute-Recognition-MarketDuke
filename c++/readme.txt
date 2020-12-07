现在目录下的文件
CMakeLists.txt
main.cpp
libtorch 找到对应CUDA版本的下载，下载好找个路径解压。


1、新建build文件夹
mkdir build
cd build

2、执行cmake /make生成文件
cmake -DCMAKE_PREFIX_PATH=../libtorch ..

3、执行cmake 得到可执行文件
cmake --build . --config Release
或者 make

