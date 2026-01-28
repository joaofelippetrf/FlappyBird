#include "../include/Agent.hpp"
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
    if(lastjump > 0) return;


    double nw = bird.position.y / screenHeight;
    double vel = bird.velocity.y / maxVel; 
    double distX = (obsX - bird.position.x) / screenWidth;
    double distY = (obsY - bird.position.y) /  screenHeight;
   

    std::vector<double> input = {nw, vel, distX, distY};
    
    if (brain.jump(input)) {
        bird.jump();
        lastjump =cooldown;
    }
}