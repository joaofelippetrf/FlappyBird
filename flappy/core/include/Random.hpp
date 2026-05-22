#pragma once
#include <random>

class Random {
public:
    static std::mt19937& get() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    static double getDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(get());
    }
    
    static bool getBoolean(){
        std::uniform_int_distribution<int> dist(0,1);
        return dist(get()) == 1;
    }

    static int getInt(int min, int max){
        std::uniform_int_distribution<int> dist(min,max);
        return dist(get());
    }

    static double getGaussian(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(get());
}
};