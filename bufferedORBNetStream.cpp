#include "bufferedORBNetStream.h"
#include <iostream>

bufferedORBNetStream::bufferedORBNetStream(int port, int bufferSize, int amortizationConstant)
    : context(1), socket(context, zmq::socket_type::rep), port(port), bufferSize(bufferSize), amortizationConstant(amortizationConstant)
{
  socket.bind("tcp://*:" + std::to_string(port));

  // Start the message consumer thread
  messageConsumerThread = std::thread(&bufferedORBNetStream::messageConsumer, this);
}

void bufferedORBNetStream::messageConsumer()
{
  float drainRatio = 0.5;
  int drainThreshold = bufferSize * drainRatio;

  while (!destroy)
  {
    // Request lock to mutex
    std::unique_lock<std::mutex> lock(bufferMutex);

    // Dequeue and pop the message
    std::string message = buffer.front();
    buffer.pop_front();

    // Check if the buffer was full
    if (bufferFull && buffer.size() < drainThreshold)
    {
      // Notify the producer that the buffer is no longer full
      bufferFull = false;
      bufferCondition.notify_one();
    }

    // Release lock to mutex
    lock.unlock();

    // Send the message
    sendFrame(message);
  }
}

/// @brief Encode the keypoints and descriptors into a string of format: frameNumber;numKeypoints;descriptor1;keypoint1;descriptor2;keypoint2;...
/// @param descriptors
/// @param keypoints
/// @param numKeypoints
/// @param frameNumber
/// @return The encoded string
std::string bufferedORBNetStream::encodeKeypoints(cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints, int numKeypoints, int frameNumber)
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

void bufferedORBNetStream::sendFrame(std::string encodedFrame)
{
  zmq::message_t request(encodedFrame.size());
  memcpy(request.data(), encodedFrame.c_str(), encodedFrame.size());
  std::cout << "sending frame .." << std::endl;

  try
  {
    zmq::message_t temp;
    socket.recv(temp, zmq::recv_flags::none);
    socket.send(request, zmq::send_flags::none);
  }
  catch (zmq::error_t e)
  {
    std::cout << "Error sending frame: " << e.what() << std::endl;
  }
}

/**
 * Appends the encoded frame to the buffer. If the buffer is full, the thread
 * waits until the buffer is no longer full.
*/
void bufferedORBNetStream::encodeAndSendFrameAsync(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors, int numKeypoints, int frameNumber)
{
  std::string encodedFrame = encodeKeypoints(descriptors, keypoints, numKeypoints, frameNumber);

  // Acquire lock to mutex
  std::unique_lock<std::mutex> lock(bufferMutex);

  // Wait until the buffer is no longer full if it is full
  if (bufferFull)
  {
    bufferCondition.wait(lock, [this] { return !bufferFull; });
  }

  // Perform amortization if there exists a message in the buffer
  if (buffer.size() > 0)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(amortizationConstant));
  }

  // Append the encoded frame to the buffer
  buffer.push_back(encodedFrame);

  // Check if the buffer is full. If yes then simply set the flag to true
  if (buffer.size() == bufferSize)
  {
    bufferFull = true;
  }

  // Lock is automatically released when the unique_lock goes out of scope.
}

bufferedORBNetStream::~bufferedORBNetStream()
{
  socket.close();
  destroy = true;
  if (messageConsumerThread.joinable())
  {
    messageConsumerThread.join();
  }
}