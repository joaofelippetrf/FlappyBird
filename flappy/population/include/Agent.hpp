#pragma once
#include "../../objects/include/Body.hpp"
#include "Brain.hpp"
#include <vector>

struct Agent {
    Body bird;
    Brain brain;
    float fitness = 0.0f;
    float lastjump = 0.0f;
    float cooldown = 0.1f;

    Agent(Brain brain);
    void act(float obsX, float obsY, double dt);
};