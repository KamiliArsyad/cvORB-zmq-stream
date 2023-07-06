#include <opencv2/core/utility.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class Benchmark {
    cv::TickMeter tm;
    std::vector<double> times;
    std::string prefix;

public:
    Benchmark(const std::string& prefix);

    void start(); 

    void set(); 

    void show(); 
    
    void show(int x); 
};

