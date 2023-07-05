#include <bitset>
#include <string>

#include <zmq.hpp>

/**
 * Object to handle all the streaming operations.
 */
class cvORBNetStream
{
  private:
    zmq::context_t context;
    zmq::socket_t socket;
    int port = 0;
  public:
    cvORBNetStream();
    ~cvORBNetStream();


    /**
     * Initialize an edge/node/worker stream.
     * @param port The port to listen to.
     * @return 0 if successful, -1 otherwise.
     */
    int Init(int port);

    /**
     * Send an encoded frame.
     * @param encodedFrame The encoded frame to send.
     * @return 0 if successful, -1 otherwise.
     */
    int SendFrame(std::string encodedFrame);
};