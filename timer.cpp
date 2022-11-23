#include "Timer.h"

Timer::Timer(const std::string &str) {
    start_time_ = std::chrono::steady_clock::now();
    str_ = str;
}

Timer::~Timer() {
    auto dur = std::chrono::steady_clock::now() - start_time_;
    std::cerr << "[" << str_ << "]: " << dur / std::chrono::microseconds (1) << "us." << std::endl;
}
