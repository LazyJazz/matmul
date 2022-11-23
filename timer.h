#pragma once
#include "string"
#include "chrono"
#include "iostream"

class Timer {
public:
    explicit Timer(const std::string& str);
    ~Timer();
private:
    std::string str_;
    std::chrono::steady_clock::time_point start_time_;
};