#pragma once
#include "../../objects/include/Body.hpp"
#include "Brain.hpp"
#include <vector>
#include <deque>

// Domain randomization (sim-to-real): definido em main.cpp, ligado por env var.
//   g_noise_std   ruído gaussiano somado aos 4 inputs (0 = desligado)
//   g_delay_frames atraso de input em frames, simula latência da câmera (0 = sem)
extern double g_noise_std;
extern int    g_delay_frames;

struct Agent {
    Body bird;
    Brain brain;
    float fitness = 0.0f;
    float lastjump = 0.0f;
    float cooldown = 0.1f;

    std::deque<std::vector<double>> inputHist;   // histórico p/ simular latência

    Agent(Brain brain);
    void act(float obsX, float obsY, double dt);
};