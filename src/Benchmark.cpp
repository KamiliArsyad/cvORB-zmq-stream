#include "Benchmark.h"

Benchmark::Benchmark(const std::string& prefix) : prefix(prefix) {}

void Benchmark::start() {
  tm.start();
}

void Benchmark::set() {
  tm.stop();
  times.push_back(tm.getTimeMilli());
  tm.reset();
}

void Benchmark::show() {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / times.size();

  std::vector<double> diff(times.size());
  std::transform(times.begin(), times.end(), diff.begin(),
                 [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / times.size());

  std::nth_element(times.begin(), times.begin() + times.size()/2, times.end());
  double median = times[times.size()/2];

  std::cout << " -------------------------------- \n"
            << "Time stats for " << prefix << ": "
            << "\nMean: " << mean << " ms"
            << "\nMedian: " << median << " ms"
            << "\nStd Dev: " << stdev << " ms" 
            << "\n ---------------------------------"<< std::endl;
}

void Benchmark::show(int x) {
  double total = std::accumulate(times.begin(), times.end(), 0.0);
  std::cout << "time for " << prefix << ": " << total / x << " ms" << std::endl;
}

