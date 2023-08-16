# CV ORB Network Stream
This is a simple high performance one-to-one network stream code for streaming extracted ORB features from each frame collected by an edge device to some server. The abstraction relies on ZeroMQ for the network communication and is written in C++.
The codes here are cloned from my previous repository of the same task using VPI 2.x and have since been improved and tuned. The ORB extraction algorithm is implemented using OpenCV 4.x. An implementation of ANMS_SSC algorithm was implemented here but not used due to an unresolved bug which makes using the algorithm not worthwhile: If ANMS_SSC is activated, the ORB BRIEF descriptor calculator will not work properly on GPU even though it works fine on CPU.

`main.cpp` is the initial client code for testing the network stream. The asynchronous message encoding and sending has also been implemented in the bufferedORBNetStream module with a buffer.
Another thread has also been implemented for loading the image in `main.cpp` to parallelize the process of loading the image, extracting the ORB features and sending the message. You should start at `main.cpp` to understand how the code works.

An additional module for benchmarking has also been implemented in `benchmark.cpp`.

## Dependencies
- OpenCV 4.x **with** CUDA support and **with** contrib modules (especially the cudafeatures2d module)
- ZeroMQ 2.5 or above
- CMake 3.5 or above

## Build
```
mkdir build
cd build
cmake ..
make
```

## Run
In the `build` folder, run the command:
```
./orb_streamer <mode> <path_to_image_folder|path_to_settings_file|path_to_video> <path_to_times_file|-|path_to_settings_file> <path_to_settings_file |-|- >
```
This will run the code. To receive the encoded descriptors, you will have to set up a ZMQ REQ TCP socket to receive the message. How the message is encoded is documented in the bufferedORBNetStream module; an example decoder implementation is available in the ARCHandler module of my ORB-SLAM2 and ORB-SLAM3 repositories. An example of the message decoder is also available in the bufferedORBNetStream module.

There are a few different modes that you can run the code in:
`0` is the image loader mode, `1` is receiving image/frame from camera input, and `2` is video mode (running from a video file). The argument description can be found if you just run the code with the mode. You can also refer to the table below for the argument description.

| Mode | First argument | Second argument | Third argument | Fourth argument |
|------|----------------|-----------------|----------------|-----------------|
| **Image Loader** | 0              | path to image folder | path to times file | path to settings file |
| **Camera Input** | 1              | path to settings file | - | - |
| **Video Loader** | 2              | path to video | path to settings file | - |

### Settings file
An example settings file is available in `./assets` right here [settings.yaml](./assets/settings.yaml).

## Benchmarking
The benchmarking module is an in-code benchmarking module that can be used to benchmark the runtime of the code by showing you the mean, median, and some percentiles of the collected runtime. You'll have to wrap your code around the pairs of `start()` and `set()` function calls to start and stop the timer. Keep in mind that the timer is not thread-safe and you'll have to use a mutex to lock the timer if you're using it in a multithreaded environment. You also need to pay attention to the overhead cost of the timer itself. The timer is implemented using `std::chrono::high_resolution_clock` and the overhead starts to show if you're measuring something that takes less than 1 microsecond to complete.

To show the result of the benchmarking, you'll have to call the `show()` function. The `show()` function will print the result of the benchmarking to the standard output. 
