#include "bufferedORBNetStream.h"
#include <iostream>

bufferedORBNetStream::bufferedORBNetStream(int port, int bufferSize, int amortizationConstant)
    : context(1), socket(context, zmq::socket_type::rep), port(port), bufferSize(bufferSize), amortizationConstant(amortizationConstant)
{
  socket.bind("tcp://*:" + std::to_string(port));

  // Start the message consumer thread
  messageConsumerThread = std::thread(&bufferedORBNetStream::messageConsumer, this);
  std::cout << "Initialized buffered streamer at port " << this->port << std::endl;
}

/**
 * The message consumer thread. Consumes the buffer and sends the messages.
*/
void bufferedORBNetStream::messageConsumer()
{
  float drainRatio = 0.75;
  int drainThreshold = bufferSize * drainRatio;

  std::cout << "Initialized new thread." << std::endl;

  for (int num = 0; !done; num++)
  {
    // Request lock to mutex
    std::unique_lock<std::mutex> lock(bufferMutex);

    /*
    // Wait until the buffer is not empty
    bufferCondition.wait(lock, [this] { return !bufferEmpty;  });
    */

    if (bufferKpts.empty()) 
    {
      std::cout << "buffer empty; waiting for data ..." << std::endl;
      bufferCondition.wait(lock, [this] { return !bufferKpts.empty();  });
    }

    // Dequeue and pop the message
    /*
    std::string message = buffer.front();
    buffer.pop_front();
    */

    cv::Mat desc = bufferDesc.front();
    bufferDesc.pop_front();

    std::vector<cv::KeyPoint> kpt = bufferKpts.front();
    bufferKpts.pop_front();


    // Check if the buffer was full
    if (bufferFull && bufferKpts.size() < drainThreshold)
    {
      // Notify the producer that the buffer is no longer full
      bufferFull = false;
      bufferCondition.notify_one();
    }

    // Release lock to mutex
    lock.unlock();

    // Send the message ---------------------------
    std::cout << "mutex unlocked. Calling sendFrame on frame " << num << " ...";
    bmSendAsync.start();
    
    // Test the new encoder
    std::vector<cv::KeyPoint> kpts_new;
    cv::Mat desc_new;

    decodeKeypoints(encodeKeypoints(desc, kpt, kpt.size(), num), desc_new, kpts_new);
    std::string message = encodeKeypoints(desc_new, kpts_new, kpts_new.size(), num);

    // std::string message = encodeKeypoints(desc, kpt, kpt.size(), num);
    sendFrame(message);
    bmSendAsync.set();
    std::cout << "done" << std::endl;
    // --------------------------------------------
    
    if (destroy && bufferKpts.empty())
    {
      std::cout << "Termination condition fulfilled" << std::endl;
      done = true;
    }
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
    ss << static_cast<int>(std::floor(keypoints[i].pt.x)) 
       << "," 
       << static_cast<int>(std::floor(keypoints[i].pt.y)) 
       << ";";
  }

  return ss.str();
}

/// @brief Much faster encoder using memcpy from the descriptors matrix directly to the string
/// @param descriptors
/// @param keypoints
/// @param frameNumber
/// @return A zmq message
zmq::message_t bufferedORBNetStream::encodeKeypoints(cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints, int frameNumber)
{
  // Create a message of size 4 + 2 + 32 * numKeypoints + 4 * numKeypoints
  // 4 bytes for frameNumber + 2 bytes for numKeypoints + 32 bytes for each descriptor + 4 bytes for each keypoint
  unsigned short numKeypoints = keypoints.size();
  int messageSize = 4 + 2 + 32 * numKeypoints + 4 * numKeypoints;
  zmq::message_t message(messageSize);

  // Encode the frame number
  memcpy(message.data(), &frameNumber, 4);

  // Encode the number of keypoints
  memcpy(static_cast<char*>(message.data()) + 4, &numKeypoints, 2);

  // Make sure the descriptors matrix is continuous
  if (!descriptors.isContinuous())
  {
    std::cout << "Descriptors matrix is not continuous" << std::endl;
    return message;
  }

  // Encode the descriptors
  memcpy(static_cast<char*>(message.data()) + 6, descriptors.data, 32 * numKeypoints);

  // Encode the keypoints
  for (int i = 0; i < numKeypoints; ++i)
  {
    // Encode the keypoint
    unsigned short x = static_cast<unsigned short>(std::floor(keypoints[i].pt.x));
    unsigned short y = static_cast<unsigned short>(std::floor(keypoints[i].pt.y));

    memcpy(static_cast<char*>(message.data()) + 6 + 32 * numKeypoints + 4 * i, &x, 2);
    memcpy(static_cast<char*>(message.data()) + 6 + 32 * numKeypoints + 4 * i + 2, &y, 2);
  }

  return message;
}

/// @brief Decode the keypoints and descriptors from the encoded zmq message
/// @param message
/// @param descriptors The output descriptors matrix
/// @param keypoints The output keypoints vector
void bufferedORBNetStream::decodeKeypoints(zmq::message_t message, cv::Mat& descriptors, std::vector<cv::KeyPoint>& keypoints)
{
  // Get the frame number
  int frameNumber;
  memcpy(&frameNumber, message.data(), 4);

  // Get the number of keypoints
  unsigned short numKeypoints;
  memcpy(&numKeypoints, static_cast<char*>(message.data()) + 4, 2);

  // Get the descriptors
  descriptors = cv::Mat(numKeypoints, 32, CV_8UC1);
  memcpy(descriptors.data, static_cast<char*>(message.data()) + 6, 32 * numKeypoints);

  // Get the keypoints
  keypoints.clear();
  for (int i = 0; i < numKeypoints; ++i)
  {
    unsigned short x, y;
    memcpy(&x, static_cast<char*>(message.data()) + 6 + 32 * numKeypoints + 4 * i, 2);
    memcpy(&y, static_cast<char*>(message.data()) + 6 + 32 * numKeypoints + 4 * i + 2, 2);

    keypoints.push_back(cv::KeyPoint(x, y, 1));
  }
}


/**
 * Send the encoded frame to the specified port. This process blocks until a request is received and the message is sent.
 * @param encodedFrame The encoded frame to send
*/
void bufferedORBNetStream::sendFrame(std::string encodedFrame)
{
  zmq::message_t request(encodedFrame.size());
  memcpy(request.data(), encodedFrame.c_str(), encodedFrame.size());

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
  // buffer.push_back(encodedFrame);
  cv::Mat desc_copy = descriptors.clone();
  bufferDesc.push_back(desc_copy);
  bufferKpts.push_back(keypoints);

  // Check if the buffer is full. If yes then simply set the flag to true
  if (bufferKpts.size() == bufferSize)
  {
    bufferFull = true;
  }

  // Notify the consumer that the buffer is no longer empty
  bufferCondition.notify_one();

  // Lock is automatically released when the unique_lock goes out of scope.
}

bufferedORBNetStream::~bufferedORBNetStream()
{
  destroy = true;
  std::cout << "terminating sending thread ..." << std::endl;

  if (messageConsumerThread.joinable())
  {
    messageConsumerThread.join();
  }
  std::cout << "Sending thread shut down" << std::endl;
  bmSendAsync.show();
  socket.close();
}
