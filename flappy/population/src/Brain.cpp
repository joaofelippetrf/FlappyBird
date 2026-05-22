#include "../include/Brain.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include "../../core/include/Random.hpp"

Brain::Brain()
{
    weights_hidden.resize(numInput * numHidden);
    for (auto &w : weights_hidden){
        w = Random::getDouble(-1.0, 1.0);
    }

    biases_hidden.resize(numHidden);
    for (auto &b : biases_hidden){
        b = Random::getDouble(-1.0, 1.0);
    }

    weights_output.resize(numHidden);
    for (auto &w : weights_output){
        w = Random::getDouble(-1.0, 1.0);
    }

    biases_output = Random::getDouble(-1.0, 1.0);
}

Brain::Brain(std::vector<double> w_h, std::vector<double> b_h, std::vector<double> w_o, double b_o)
    : weights_hidden(std::move(w_h)),
      biases_hidden(std::move(b_h)),
      weights_output(std::move(w_o)),
      biases_output(b_o)
{
}

double Brain::tanh(double x) const{
    return std::tanh(x);
}

bool Brain::jump(const std::vector<double> &input){
    std::vector<double> hidden(numHidden);

    for (int j = 0; j < numHidden; j++)
    {
        double sum = biases_hidden[j];
        for (int i = 0; i < numInput; i++)
        {
            sum += input[i] * weights_hidden[j * numInput + i];
        }
        hidden[j] = tanh(sum);
    }

    double output_sum = biases_output;
    for (int i = 0; i < numHidden; i++)
    {
        output_sum += hidden[i] * weights_output[i];
    }

    return tanh(output_sum) >= 0.0;
}

double Brain :: addMutation(double mutationRate, double mutationStep){
    double ans = 0;
    if (Random::getDouble(0.0, 1.0) < mutationRate) {
        ans += Random::getGaussian(0.0, mutationStep);
    }
    return ans;
}

std :: vector<double> Brain :: vecFromUniform(const std::vector<double>& fatherInput,const std::vector<double>& motherInput, double mutationRate, double mutationStep){
    bool takeItFromFather;
    int size = fatherInput.size();
    std :: vector<double> output(size);

    for(int j = 0; j < size; j++){
        takeItFromFather = Random::getBoolean() ; 
        if(takeItFromFather) output[j] = fatherInput[j];
        else output[j] = motherInput[j];

        output[j] += addMutation(mutationRate, mutationStep);
    }
    return output;
}

std :: vector<double> Brain :: vecFromArithmetic(const std::vector<double>& fatherInput,const std::vector<double>& motherInput, double mutationRate, double mutationStep){
    int size = fatherInput.size();
    std :: vector<double> output(size);

    for(int j = 0; j < size; j++){
        output[j] = fatherInput[j]  + motherInput[j];
        output[j] /= 2.0;

        output[j] += addMutation(mutationRate, mutationStep);
    }
    return output;
}

Brain Brain :: makeChildUniform (const Brain& mother){
    double mutationRate = 0.05; 
    double mutationStep = 0.1;

    auto wh = vecFromUniform(this->weights_hidden, mother.weights_hidden, mutationRate, mutationStep);
    auto bh = vecFromUniform(this->biases_hidden, mother.biases_hidden, mutationRate, mutationStep);
    auto wo = vecFromUniform(this->weights_output, mother.weights_output, mutationRate, mutationStep);
    auto boTemp = vecFromUniform({this->biases_output},{mother.biases_output}, mutationRate, mutationStep);
    double bo = boTemp[0];
    
    return Brain(std :: move(wh),std :: move (bh), std :: move(wo), bo);  
}

Brain Brain :: makeChildArithmetic (const Brain& mother){
    double mutationRate = 0.05; 
    double mutationStep = 0.1;

    auto wh = vecFromArithmetic(this->weights_hidden, mother.weights_hidden, mutationRate, mutationStep);
    auto bh = vecFromArithmetic(this->biases_hidden, mother.biases_hidden, mutationRate, mutationStep);
    auto wo = vecFromArithmetic(this->weights_output, mother.weights_output, mutationRate, mutationStep);
    auto boTemp = vecFromArithmetic({this->biases_output},{mother.biases_output}, mutationRate, mutationStep);
    double bo = boTemp[0];
    
    return Brain(std :: move(wh),std :: move (bh), std :: move(wo), bo); 
}