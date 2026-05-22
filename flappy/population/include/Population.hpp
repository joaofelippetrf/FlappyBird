#pragma once
#include "../../objects/include/Body.hpp"
#include "Agent.hpp"
#include <vector>

struct Population {
    int gen = 1;
    std:: vector<Agent> agents;    
    int initialBirds;
    Population(int initialBirds);
    int runTournament(const std::vector<Agent> &agents, int size);
    void getEliteChildren(int& idx, int parents);
    void getDiverseChildren(int idx, int size);

    void sort();
    void restartBirds();
    void updateScore();

};