#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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
  bool useImage = false;

  // The buffer is a deque of encoded frames. Message is pushed to the back and
  // popped from the front.
  std::deque<std::string> buffer;
  std::deque<cv::Mat> bufferDesc;
  std::deque<std::vector<cv::KeyPoint>> bufferKpts;
  std::deque<double> bufferTimestamps;

  // Image support
  std::deque<cv::Mat> bufferImg;

  /**
   * Thread to consume the buffer and send the messages. This thread keeps running
   * until the program is terminated or the object is destroyed.
   */
  std::thread messageConsumerThread;

  // The conditional variable and mutex to synchronize the buffer.
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

  zmq::message_t encodeKeypoints(cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints, int frameNumber, double timestamp=0);

  zmq::message_t encodeKeypoints(cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints, int frameNumber, cv::Mat img, double timestamp=0);

  void decodeKeypoints(zmq::message_t message, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints);

  /**
   * Thread to consume the buffer and send the messages. This thread keeps running
   * until the program is terminated or the object is destroyed.
   */
  void messageConsumer();

public:
  /**
   * Initialize an edge/node/worker stream.
   * @param port The port to listen to.
   * @param bufferSize The size of the buffer.
   * @param amortizationConstant The number of milliseconds to hold the calling thread before
   *                              sending the next message if there exists a message in the buffer.
   * @param useImage Whether to use image or not.
   */
  bufferedORBNetStream(int port, int bufferSize, int amortizationConstant, bool useImage=false);

  /**
   * Destructor. Joins the message consumer thread. This might take a while as the thread empties the buffer if it's not empty.
   */
  ~bufferedORBNetStream();

  /**
   * Send an encoded frame.
   * @param encodedFrame The encoded frame to send.
   */
  void sendFrame(std::string encodedFrame);

  /**
   * Send an encoded frame (zmq::message_t version).
   * @param encodedFrame The encoded frame to send.
   */
  void sendFrame(zmq::message_t encodedFrame);

  void encodeAndSendFrame(std::vector<KeyPoint> keypointsArray,
                          Mat descriptorsArray, int numKeypoints,
                          int frameNumber);

  /**
   * Appends the encoded frame to the buffer to be consumed by the messageConsumerThread. The message consumer thread automatically consumes the buffer and sends the messages.
   * @param keypoints The keypoints to encode and send.
   * @param descriptors The descriptors to encode and send.
   * @param numKeypoints The number of keypoints.
   * @param frameNumber The frame number.
   */
  void encodeAndSendFrameAsync(std::vector<KeyPoint> keypoints,
                               Mat descriptors, int numKeypoints,
                               int frameNumber, double timestamp=0);

  /**
   * Appends the encoded frame (with image) to the buffer to be consumed by the messageConsumerThread. The message consumer thread automatically consumes the buffer and sends the messages.
   * @param keypoints The keypoints to encode and send.
   * @param descriptors The descriptors to encode and send.
   * @param frameNumber The frame number.
   * @param img The image to encode and send.
   * @param timestamp The timestamp of the frame.
   */
  void encodeAndSendFrameAsync(std::vector<KeyPoint> keypoints,
                               Mat descriptors, int frameNumber, Mat img, double timestamp);
};
