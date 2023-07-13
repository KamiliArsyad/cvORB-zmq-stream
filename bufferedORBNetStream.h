#include <opencv2/core.hpp>
#include <bitset>
#include <string>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "./Benchmark.h"

#include <zmq.hpp>

using namespace cv;

/**
 * Object to handle all the streaming operations.
 */
class bufferedORBNetStream
{
private:
  zmq::context_t context;
  zmq::socket_t socket;
  int port;

  int bufferSize = 100;
  int amortizationConstant = 10; // The number of milliseconds to wait before sending the next message if there exists a message in the buffer.

  // The buffer is a deque of encoded frames. Message is pushed to the back and
  // popped from the front.
  std::deque<std::string> buffer;
  std::deque<cv::Mat> bufferDesc;
  std::deque<std::vector<cv::KeyPoint>> bufferKpts;

  /**
   * Thread to consume the buffer and send the messages. This thread keeps running
   * until the program is terminated or the object is destroyed.
  */
  std::thread messageConsumerThread;

  // The condition variable and mutex to synchronize the buffer.
  std::condition_variable bufferCondition;
  std::mutex bufferMutex;
  bool bufferEmpty = true;
  bool bufferFull = false;
  bool destroy = false;
  bool done = false;

  /**
   * Benchmarking variables. Not all of them are used.
  */
  Benchmark bmSendAsync = Benchmark("calling send async");
  Benchmark bmEncode = Benchmark("encoding the frame");
  Benchmark bmSendAsyncLock = Benchmark("acquiring lock");
  Benchmark bmEncodeDescriptor = Benchmark("encoding one 256-bit descriptor");
  Benchmark bmEncodeKeypoint = Benchmark("encoding one keypoint");

  /**
   * Encode the keypoints and their respective descriptors of a frame
   * into a single string. The format is as follows:
   * <frameNumber>;<numKeypoints>;<desc1>;<x1>,<y1>;<desc2>;<x2>,<y2>;...
   * The descriptors are encoded as 32 characters ASCII strings for efficiency.
   */
  std::string encodeKeypoints(Mat descriptors, std::vector<KeyPoint> keypoints, 
                              int numKeypoints, int frameNumber);

  void messageConsumer();

public:
  /**
   * Initialize an edge/node/worker stream.
   * @param port The port to listen to.
   * @param bufferSize The size of the buffer.
   * @param amortizationConstant The number of milliseconds to hold the calling thread before 
   *                              sending the next message if there exists a message in the buffer.
  */
  bufferedORBNetStream(int port, int bufferSize, int amortizationConstant);
  ~bufferedORBNetStream();

  /**
   * Send an encoded frame.
   * @param encodedFrame The encoded frame to send.
   */
  void sendFrame(std::string encodedFrame);

  void encodeAndSendFrame(std::vector<KeyPoint> keypointsArray,
                         Mat descriptorsArray, int numKeypoints,
                         int frameNumber);
  
  void encodeAndSendFrameAsync(std::vector<KeyPoint> keypoints,
                               Mat descriptors, int numKeypoints,
                               int frameNumber);
};
