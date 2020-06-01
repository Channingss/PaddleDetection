OPENCV_URL=https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
if [ ! -d "./deps/opencv3.4.6gcc4.8ffmpeg.tar.gz2" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv3.4.6gcc4.8ffmpeg.tar.gz2
    cd ..
fi
