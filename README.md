# CV ORB Network Stream
This is a simple high performance one-to-one network stream code for streaming extracted ORB features from each frame collected by an edge device to some server. The abstraction relies on ZeroMQ for the network communication and is written in C++.
The codes here are cloned from my previous repository of the same task using VPI 2.x and have since been improved and tuned. The ORB extraction algorithm is implemented using OpenCV 4.x. An implementation of ANMS_SSC algorithm was implemented here but not used due to an unresolved bug which makes using the algorithm not worthwhile: If ANMS_SSC is activated, the ORB BRIEF descriptor calculator will not work properly on GPU even though it works fine on CPU.

`main.cpp` is the initial client code for testing the network stream. The asynchronous message encoding and sending has also been implemented in the bufferedORBNetStream module with a buffer.
Another thread has also been implemented for loading the image in `main.cpp` to parallelize the process of loading the image, extracting the ORB features and sending the message.

An additional module for benchmarking has also been implemented in `benchmark.cpp`.

The next step is to standardize the parameter adjustment in an appropriate settings file and perhaps reconfigure/tune the ZMQ socket to ignore the cost of sending the message, and perhaps to find a more efficient way to encode the message. Currently, encoding the message for 2500 features takes about 10ms on a Jetson Nano which is not ideal.

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
```
cd build
./cv_orb_zmq_stream <path_to_image_folder> <path_to_timestamp_file>
```
This will run the code. To receive the encoded descriptors, you will have to set up a ZMQ REQ TCP socket to receive the message. How the message is encoded is documented in the bufferedORBNetStream module; an example decoder implementation is available in my ORB-SLAM2 repository in the ARCHandler module.