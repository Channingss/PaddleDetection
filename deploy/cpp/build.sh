rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=ON \
    -DWITH_MKL=ON \
    -DWITH_TENSORRT=OFF \
    -DPADDLE_DIR=/chenlingchi/docker/lib/fluid_inference/ \
    -DCUDA_LIB=/usr/local/cuda/lib64/ \
    -DCUDNN_LIB=/usr/lib/x86_64-linux-gnu/ \
    -DOPENCV_DIR=/chenlingchi/docker/lib/opencv-builded-3.4.6/
make -j24
