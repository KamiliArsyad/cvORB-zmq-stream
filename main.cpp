/**
 * This code is made for monocular eurocc dataset. No setting file is needed or used as of right now as we
 * are only ensure that the proposed model is working.
 */
#include "./cvORBNetStream.h"
#include "./bufferedORBNetStream.h"

// #include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cstring> // for memset
#include <iostream>
#include <bitset>
#include <fstream>
#include <numeric>
#include <condition_variable>
#include <thread>
#include <mutex>

#define DEMO_MODE 1 // Demo mode will send the images to the server as well

#define IMAGE_LOADER_MODE 0 // Image loader mode will load the images from the given dataset
#define CAMERA_MODE 1       // Camera mode will use the camera as the source of images
#define VIDEO_MODE 2        // Video mode will use the specified video as the source of images

using namespace cv;

/**
 * @brief Adaptive Non-Maximal Suppression (ANMS) using Supresion via Square Covering (SSC).
 *        The algorithm is based on the paper "Efficient adaptive non-maximal suppression algorithms
 *        for homogeneous spatial keypoint distribution" by Bailo, et al. (2018). The code is sourced
 *        from the repository referenced in the paper with some minor modifications (if any).
 * @param unsortedKeypoints The keypoints to be sorted and filtered.
 * @param numRetPoints The maximum number of keypoints to be returned.
 * @param tolerance The tolerance parameter for the SSC algorithm.
 * @param cols The number of columns in the image.
 * @param rows The number of rows in the image.  */
std::vector<KeyPoint> ANMS_SSC(std::vector<KeyPoint> unsortedKeypoints, int numRetPoints, float tolerance, int cols, int rows);

void LoadImages(const std::string &strImagePath, const std::string &strPathTimes,
                std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps);

/**
 * First argument: The mode of operation (0: image loader, 1: camera, 2: video)
 * Second argument: The path to the image folder (if mode is 0), or the path to settings file (if mode is 1), or the path to the video (if mode is 2)
 * Third argument: The path to the times file (if mode is 0) or the path to the settings file (if mode is 2)
 * Fourth argument: The path to the settings file (if mode is 0)
 *
 * Arguments table
 *    Mode     |   First argument  |   Second argument       |   Third argument       |   Fourth argument
 *    0        |   0               |   path to image folder  |   path to times file   |  path to settings file
 *    1        |   1               |   path to settings file |    -                   |   -
 *    2        |   2               |   path to video         |   path to settings file|   -
 */
int main(int argc, char *argv[])
{
  int returnValue = 0;

  if (argc < 2)
  {
    throw std::runtime_error(std::string("Usage: ") + argv[0] + " <mode> <path_to_image_folder|path_to_settings_file|path_to_video> <path_to_times_file|-|path_to_settings_file> <path_to_settings_file|-|->");
  }

  std::vector<std::string> vstrImageFilenames;
  std::vector<double> vTimestamps;
  int mode = atoi(argv[1]);

  std::cout << "Mode choosen is " << std::string(mode == IMAGE_LOADER_MODE ? "IMAGE_LOADER_MODE" : mode == CAMERA_MODE ? "CAMERA_MODE"
                                                                                                                       : "VIDEO_MODE")
            << std::endl;

  if (mode == IMAGE_LOADER_MODE)
  {
    if (argc != 5)
    {
      throw std::runtime_error(std::string("Usage: ") + argv[0] + " 0 <path_to_image_folder> <path_to_times_file> <path_to_settings_file>");
    }
  }
  else if (mode == CAMERA_MODE)
  {
    if (argc != 3)
    {
      throw std::runtime_error(std::string("Usage: ") + argv[0] + " 1 <path_to_settings_file>");
    }
  }
  else if (mode == VIDEO_MODE)
  {
    if (argc != 4)
    {
      throw std::runtime_error(std::string("Usage: ") + argv[0] + " 2 <path_to_video> <path_to_settings_file>");
    }
  }
  else
  {
    throw std::runtime_error(std::string("Usage: ") + argv[0] + " <mode> <path_to_image_folder|path_to_settings_file|path_to_video> <path_to_times_file|-|path_to_settings_file>");
  }

  // MAX INTEGER
  int numOfFrames = std::numeric_limits<int>::max();

  if (mode = IMAGE_LOADER_MODE)
  {
    // Load images and timestamps
    LoadImages(argv[2], argv[3], vstrImageFilenames, vTimestamps);

    numOfFrames = vstrImageFilenames.size();

    if (numOfFrames <= 0)
    {
      throw std::runtime_error("Couldn't load images");
    }
  }

  // ---------------------
  // Load the settings file
  // ---------------------
  int argcSettings = 0;
  argcSettings = argc - 1;
  
  cv::FileStorage fsSettings(argv[argcSettings], cv::FileStorage::READ);

  // Check
  if (!fsSettings.isOpened())
  {
    throw std::runtime_error("Could not open settings file");
  }

  std::cout << "Settings file loaded" << std::endl;

  // ---------------------
  // Limit the number of frames to whatever is set in the settings file
  // ---------------------
  int maxFrames = fsSettings["maxFrames"];
  numOfFrames = std::min(numOfFrames, maxFrames);

  //      CV ORB creation
  //      ---------------------
  std::vector<KeyPoint> keypoints;
  std::vector<KeyPoint> filteredKeypoints;
  cuda::GpuMat descriptors;
  int nFeatures = fsSettings["ORBExtractor.nFeatures"];
  float fScaleFactor = fsSettings["ORBExtractor.scaleFactor"];
  int nLevels = fsSettings["ORBExtractor.nLevels"];
  int fIniThFAST = fsSettings["ORBExtractor.iniThFAST"];
  int blurForDescriptor = fsSettings["ORBExtractor.blurForDescriptor"];
  Mat descriptorsCPU;

  Ptr<cuda::ORB> orb = cuda::ORB::create(nFeatures, fScaleFactor, nLevels, 31, 0, 2, ORB::HARRIS_SCORE, 31, fIniThFAST, blurForDescriptor);

  //      ---------------------
  //      UNUSED
  //      ---------------------
  int numKeyPoints = 0;
  Ptr<cuda::FastFeatureDetector> fastDetector = cuda::FastFeatureDetector::create(15);
  fastDetector->setMaxNumPoints(nFeatures);
  Ptr<cv::ORB> orbCPU = cv::ORB::create(nFeatures);
  //      ---------------------

  cv::Mat frame;
  cuda::GpuMat frameGPU(frame);

  // Setup a thread and its corresponding synchronization mechanism for loading the images
  std::mutex mtxImageLoader;
  std::condition_variable convarImageLoader;
  bool ready = false;

  std::thread image_loading_thread([&]()
                                   {
                        
    cv::VideoCapture vid;
    if (mode = CAMERA_MODE)
    {
      vid.open("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1080, height=(int)720,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink");
      if (!vid.isOpened()) 
      {
        std::cerr << "ERROR! Unable to open camera\n";
      }
    } else if (mode = VIDEO_MODE)
    {
      vid.open(argv[2]);
      if (!vid.isOpened()) 
      {
        std::cerr << "ERROR! Unable to open video\n";
      }
    }

    for (int i = 0; i < numOfFrames; i++)
    {
      cv::Mat new_frame;

      if (mode = IMAGE_LOADER_MODE)
      {
        new_frame = cv::imread(vstrImageFilenames[i], cv::IMREAD_UNCHANGED);
      } else
      {
        cv::Mat new_frame_color;
        vid >> new_frame_color;

        cv::cvtColor(new_frame_color, new_frame, cv::COLOR_BGR2GRAY);
      }

      // wait until the main thread is ready for the next image
      std::unique_lock<std::mutex> lock(mtxImageLoader);
      convarImageLoader.wait(lock, [&]() { return ready; });

      // main thread is ready
      frame = new_frame;
      ready = false;

      // Notify
      convarImageLoader.notify_one();
    } });

  bufferedORBNetStream bufferedORBStream(fsSettings["bufferedORBNetStream.port"], fsSettings["bufferedORBNetStream.bufferSize"], fsSettings["bufferedORBNetStream.amortizationDelay"], DEMO_MODE);

  Benchmark bmTotal("computing each frame");
  Benchmark bmORB("detecting and computing ORB descriptors");
  Benchmark bmSend("sending frames");

  // Process each frame
  for (int i = 0; i < numOfFrames; i++)
  {
    bmTotal.start();
    printf("processing frame %d\n", i);
    // Image reading -----------------------------------------------
    // Indicate that we're ready for the next image
    {
      std::lock_guard<std::mutex> lock(mtxImageLoader);
      ready = true;
    }
    convarImageLoader.notify_one();

    // Wait until the image loading thread gives us the next image
    std::unique_lock<std::mutex> lock(mtxImageLoader);
    convarImageLoader.wait(lock, [&]()
                           { return !ready; });
    // reading done -------------------------------------------------

    // ---------------------
    // Process the frame
    // ---------------------
    frameGPU.upload(frame);

    bmORB.start();
    orb->detect(frameGPU, keypoints);

    // Unused ANMS_SSC
    // filteredKeypoints = ANMS_SSC(keypoints, 1000, 0.1, frame.cols, frame.rows);
    // orbCPU->compute(frame, filteredKeypoints, descriptorsCPU);
    filteredKeypoints = keypoints;

    orb->compute(frameGPU, filteredKeypoints, descriptors);
    bmORB.set();

    descriptors.download(descriptorsCPU);

    bmSend.start();
    if (DEMO_MODE)
      bufferedORBStream.encodeAndSendFrameAsync(filteredKeypoints, descriptorsCPU, i, frame);
    else
      bufferedORBStream.encodeAndSendFrameAsync(filteredKeypoints, descriptorsCPU, filteredKeypoints.size(), i);
    bmSend.set();
    bmTotal.set();
  }

  image_loading_thread.join();

  bmORB.show();
  bmSend.show();
  bmTotal.show();

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

std::vector<KeyPoint> ANMS_SSC(std::vector<KeyPoint> unsortedKeypoints, int numRetPoints, float tolerance, int cols, int rows)
{
  // Sort the keypoints by decreasing strength
  std::vector<float> responseVector;
  for (unsigned int i = 0; i < unsortedKeypoints.size(); i++)
    responseVector.push_back(unsortedKeypoints[i].response);
  std::vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  sortIdx(responseVector, Indx, cv::SORT_DESCENDING);

  // The sorted keypoints
  std::vector<KeyPoint> keypoints;
  for (unsigned int i = 0; i < unsortedKeypoints.size(); i++)
    keypoints.push_back(unsortedKeypoints[Indx[i]]);

  // several temp expression variables to simplify solution equation
  int exp1 = rows + cols + 2 * numRetPoints;
  long long exp2 =
      ((long long)4 * cols + (long long)4 * numRetPoints +
       (long long)4 * rows * numRetPoints + (long long)rows * rows +
       (long long)cols * cols - (long long)2 * rows * cols +
       (long long)4 * rows * cols * numRetPoints);
  double exp3 = sqrt(exp2);
  double exp4 = numRetPoints - 1;

  double sol1 = -round((exp1 + exp3) / exp4); // first solution
  double sol2 = -round((exp1 - exp3) / exp4); // second solution

  // binary search range initialization with positive solution
  int high = (sol1 > sol2) ? sol1 : sol2;
  int low = floor(sqrt((double)keypoints.size() / numRetPoints));
  low = max(1, low);

  int width;
  int prevWidth = -1;

  std::vector<int> ResultVec;
  bool complete = false;
  unsigned int K = numRetPoints;
  unsigned int Kmin = round(K - (K * tolerance));
  unsigned int Kmax = round(K + (K * tolerance));

  std::vector<int> result;
  result.reserve(keypoints.size());
  while (!complete)
  {
    width = low + (high - low) / 2;
    if (width == prevWidth ||
        low >
            high)
    {                     // needed to reassure the same radius is not repeated again
      ResultVec = result; // return the keypoints from the previous iteration
      break;
    }
    result.clear();
    double c = (double)width / 2.0; // initializing Grid
    int numCellCols = floor(cols / c);
    int numCellRows = floor(rows / c);
    std::vector<std::vector<bool>> coveredVec(numCellRows + 1,
                                              std::vector<bool>(numCellCols + 1, false));

    for (unsigned int i = 0; i < keypoints.size(); ++i)
    {
      int row =
          floor(keypoints[i].pt.y /
                c); // get position of the cell current point is located at
      int col = floor(keypoints[i].pt.x / c);
      if (coveredVec[row][col] == false)
      { // if the cell is not covered
        result.push_back(i);
        int rowMin = ((row - floor(width / c)) >= 0)
                         ? (row - floor(width / c))
                         : 0; // get range which current radius is covering
        int rowMax = ((row + floor(width / c)) <= numCellRows)
                         ? (row + floor(width / c))
                         : numCellRows;
        int colMin =
            ((col - floor(width / c)) >= 0) ? (col - floor(width / c)) : 0;
        int colMax = ((col + floor(width / c)) <= numCellCols)
                         ? (col + floor(width / c))
                         : numCellCols;
        for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov)
        {
          for (int colToCov = colMin; colToCov <= colMax; ++colToCov)
          {
            if (!coveredVec[rowToCov][colToCov])
              coveredVec[rowToCov][colToCov] =
                  true; // cover cells within the square bounding box with width
                        // w
          }
        }
      }
    }

    if (result.size() >= Kmin && result.size() <= Kmax)
    { // solution found
      ResultVec = result;
      complete = true;
    }
    else if (result.size() < Kmin)
      high = width - 1; // update binary search range
    else
      low = width + 1;
    prevWidth = width;
  }
  // retrieve final keypoints
  std::vector<KeyPoint> kp;
  for (unsigned int i = 0; i < ResultVec.size(); i++)
    kp.push_back(keypoints[ResultVec[i]]);

  return kp;
}
