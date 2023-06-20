#include "cvORBNetStream.h"
#include <iostream>
#include <zmq.hpp>

int main() 
{
  cvORBNetStream stream;
  stream.Init("localhost", 9999);

  while (true) {
    FrameData frameData = stream.ReceiveFrame();
    std::cout << "Received frame " << frameData.frameNumber << std::endl;
  }
}