#include "../include/Population.hpp"
#include "../include/Brain.hpp"
#include "../../core/include/Random.hpp"
#include "../../objects/include/Body.hpp"
#include <vector>
#include <iostream>
#include <algorithm>



Population :: Population(int initialBirds) : initialBirds(initialBirds)
{
    agents.reserve(initialBirds);
    for(int i =0; i<initialBirds;i++){
        agents.emplace_back(Brain());
    }
}

int Population :: runTournament(const std::vector<Agent> &agents, int size){
    int best = Random::getInt(0, initialBirds -1);
    for (int i = 1; i < size; i++)
    {
        int candidate = Random::getInt(0, initialBirds -1);
        if (agents[candidate].fitness > agents[best].fitness)
        {
            best = candidate;
        }
    }
    return best;
}

void Population :: sort(){
        std::sort(agents.begin(), agents.end(), [](const Agent &a, const Agent &b)
            { return a.fitness > b.fitness; });
}

void Population :: updateScore(){
    for (auto &agent : agents){
        if (agent.bird.alive){
            agent.bird.score++;
        }
    }
}

void Population :: getEliteChildren(int& idx, int parents){
        for (int i = 0; i < parents; i++){
            for (int j = i + 1; j < parents; j++){
                agents[idx].brain = agents[i].brain.makeChildArithmetic(agents[j].brain);
                idx++;
            }
    }
}

void Population :: getDiverseChildren(int idx, int size){
    for (int i = idx; i < initialBirds; i++){
        if (Random::getDouble(0, 1) < 0.95){
            int p1 = runTournament(agents, size);
            int p2 = runTournament(agents, size);
            agents[i].brain = agents[p1].brain.makeChildUniform(agents[p2].brain);
        }
        else{
            agents[i].brain = Brain();
        }
    }
}

void Population :: restartBirds(){
    for(auto& agent : agents){
        agent.bird = Body({200, 540}, {0,0});
        agent.fitness =0.0f;
        agent.lastjump =0.0f;
    }
}