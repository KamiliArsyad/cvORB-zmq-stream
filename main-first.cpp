#include "./cvORBNetStream.h"

#include <opencv2/core/version.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <bitset>
#include <iostream>
#include <zmq.hpp>
#define DEBUG true

using namespace cv::cuda;

void LoadImages(const std::string &strImagePath, const std::string &strPathTimes,
                std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps);


/// @brief Encode the keypoints and descriptors into a string of format: frameNumber;numKeypoints;descriptor1;keypoint1;descriptor2;keypoint2;...
/// @param descriptors
/// @param keypoints 
/// @param numKeypoints 
/// @param frameNumber 
/// @return The encoded string
std::string encoder(cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints, int numKeypoints, int frameNumber)
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
    std::vector<cv::KeyPoint> keypoints;
    GpuMat descriptors;
    cv::Mat descriptorsCPU;
    // cv::Mat descriptors;
    int nFeatures = 10000;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::Ptr<ORB> orb = ORB::create(nFeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20, true);
    // cv::Ptr<ORB> orb = cv::cuda::ORB::create(nFeatures);
    //      ---------------------

    // Initialize a timer
    cv::TickMeter timer;
    timer.start();

    cv::Mat frame, frameGray;
    inputCamera >> frame; // Fetch a new frame from camera.
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    GpuMat frameGPU(frameGray);

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
        // orb->detectAndCompute(frame, cv::Mat(), keypoints, descriptors);
	cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        frameGPU.upload(frameGray);
	cv::TickMeter t;
	t.start();
        orb->detectAndCompute(frameGPU, GpuMat(), keypoints, descriptors);
        // orb->detect(frameGPU, keypoints);
	t.stop();
        int numKeypoints = keypoints.size();
        // ---------------------
        descriptors.download(descriptorsCPU);

        std::cout << "Number of keypoints: " << numKeypoints << std::endl;
	std::cout << "Processing time: " << t.getTimeMilli() << std::endl;
	std::cout << " ---------------------------------- " << std::endl;
        // Encode and Send
        netStream.SendFrame(encoder(descriptorsCPU, keypoints, numKeypoints, i));
    }

    // Stop the timer
    timer.stop();
    printf("Processing time per frame: %f ms\n", timer.getTimeMilli() / numOfFrames);

    // Cleanup
    inputCamera.release();

    return returnValue;
}

void LoadImages(const std::string &strImagePath, const std::string &strPathTimes,
                std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps)
{
  std::ifstream fTimes;
  fTimes.open(strPathTimes.c_str());
  vTimeStamps.reserve(5000);
  vstrImages.reserve(5000);
  while (!fTimes.eof())
  {
    std::string s;
    getline(fTimes, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;
      vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
      double t;
      ss >> t;
      vTimeStamps.push_back(t / 1e9);
    }
  }
}