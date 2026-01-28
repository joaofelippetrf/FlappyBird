#pragma once
#include <vector>

struct Brain
{
    int numInput = 4;
    int numHidden = 5;

    std::vector<double> weights_hidden;
    std::vector<double> biases_hidden;
    std::vector<double> weights_output;
    double biases_output;

    Brain();
    Brain(std::vector<double> w_h, std::vector<double> b_h, std::vector<double> w_o, double b_o);

    double tanh(double x) const;
    bool jump(const std::vector<double> &input);
    
    Brain makeChildUniform (const Brain& mother);
    Brain makeChildArithmetic (const Brain& mother);

    double addMutation (double mutationRate, double mutationStep);
    std :: vector<double> vecFromUniform(const std::vector<double>& fatherInput,const std::vector<double>& momInput, double mutationRate, double mutationStep);
    std :: vector<double> vecFromArithmetic(const std::vector<double>& fatherInput,const std::vector<double>& momInput, double mutationRate, double mutationStep);
};