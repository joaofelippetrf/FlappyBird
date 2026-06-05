#include "../include/Agent.hpp"
#include "../../core/include/Random.hpp"
#include <vector>
#include <iostream>
#include <cmath>



Agent::Agent(Brain brain) : brain(brain) , bird(Body({200, 540}, {0,0}))
{
}

void Agent::act(float obsX, float obsY, double dt) {
    lastjump -= dt;
    double screenHeight = 1080;
    double screenWidth = 500;
    double maxVel = 600;

    if (!bird.alive){
        if(fitness != 0) return;
        float heightPenalty = std::abs(obsY - bird.position.y);
        
        if(bird.score == 0){
            fitness = bird.time*bird.time;
            return;
        }
        fitness = 50*bird.score - heightPenalty/10;
        return;
    }
    double nw = bird.position.y / screenHeight;
    double vel = bird.velocity.y / maxVel;
    double distX = (obsX - bird.position.x) / screenWidth;
    double distY = (obsY - bird.position.y) /  screenHeight;

    std::vector<double> input = {nw, vel, distX, distY};

    // Latência: o histórico avança todo frame; o cérebro decide com o estado
    // de g_delay_frames atrás (simula o atraso câmera→processamento→UDP).
    inputHist.push_back(input);
    while ((int)inputHist.size() > g_delay_frames + 1) inputHist.pop_front();

    if(lastjump > 0) return;   // cooldown: não decide agora (histórico já avançou)

    std::vector<double> used = inputHist.front();

    // Ruído: simula o tremor/imprecisão da estimativa da câmera.
    if (g_noise_std > 0.0)
        for (double& v : used) v += Random::getGaussian(0.0, g_noise_std);

    if (brain.jump(used)) {
        bird.jump();
        lastjump =cooldown;
    }
}