#include "./cvORBNetStream.h"

#include <opencv2/core/version.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <bitset>
#include <cstdio>
#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>
#define DEBUG 1

using namespace cv;

/// @brief Encode the keypoints and descriptors into a string of format: frameNumber;numKeypoints;descriptor1;keypoint1;descriptor2;keypoint2;...
/// @param descriptors
/// @param keypoints 
/// @param numKeypoints 
/// @param frameNumber 
/// @return The encoded string
std::string encoder(cv::Mat descriptors, std::vector<KeyPoint> keypoints, int numKeypoints, int frameNumber)
{
    std::ostringstream ss;
    ss << frameNumber << ";" << numKeypoints << ";";

    // Encode the keypoints and descriptors
    for (int i = 0; i < numKeypoints; ++i)
    {
        // Encode the descriptor to 32 characters
        for (int j = 0; j < 32; ++j)
        {
            unsigned char byte = descriptors.at<unsigned char>(i, j);

            ss << byte;
        } 

        ss << ";";

        // Encode the keypoint
        ss << keypoints[i].pt.x << "," << keypoints[i].pt.y << ";";
    }

    return ss.str();
}

/**
 * First argument: backend (<cpu|cuda>)
 * Second argument: how many frames are going to be recorded
 */
int main(int argc, char *argv[])
{
    int returnValue = 0;

    // Parse parameters
    if (argc != 3)
    {
        throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda> <number of frames>");
    }

    int numOfFrames = std::stoi(argv[2]);

// ========================
// Process frame by frame
#if DEBUG
    cv::VideoCapture inputCamera("../assets/input.mp4");
#else
    cv::VideoCapture inputCamera(0);
#endif

    if (!inputCamera.isOpened())
    {
        throw std::runtime_error("Can't open camera\n");
        return -1;
    }

    //      CV ORB creation
    //      ---------------------
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    int nFeatures = 1000;
    Ptr<ORB> orb = ORB::create(nFeatures, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    //      ---------------------

    // Initialize a timer
    cv::TickMeter timer;
    timer.start();

    cv::Mat frame;
    inputCamera >> frame; // Fetch a new frame from camera.

    // Setup a worker stream
    cvORBNetStream netStream;
    netStream.Init(9999);

    // Process each frame
    for (int i = 0; i < numOfFrames; ++i)
    {
        printf("processing frame %d\n", i);
        inputCamera >> frame; // Fetch a new frame from camera.

        // ---------------------
        // Process the frame
        // ---------------------
        orb->detectAndCompute(frame, Mat(), keypoints, descriptors);
        int numKeypoints = keypoints.size();

        std::cout << "Number of keypoints: " << numKeypoints << std::endl;
        // Encode and Send
        netStream.SendFrame(encoder(descriptors, keypoints, numKeypoints, i));
    }

    // Stop the timer
    timer.stop();
    printf("Processing time per frame: %f ms\n", timer.getTimeMilli() / numOfFrames);

    // Cleanup
    inputCamera.release();

    return returnValue;
}