// train_headless.cpp
//
// Treinador SEM janela (sem SFML RenderWindow): roda a mesma física/evolução
// do jogo, mas sem renderizar e sem limite de 60fps — então treina MUITO mais
// rápido e roda sem display. Gera/atualiza o best_brain.txt.
//
// Compilar (linka SFML só porque World/Obstacle/Body têm membros sf::*, mas
// nunca abrimos janela nem desenhamos):
//   ver build via CMake (alvo TrainHeadless) ou g++ manual no fim do arquivo.
//
// Uso: ./train_headless [geracoes] [model_path]

#include "core/include/World.hpp"
#include "population/include/Agent.hpp"
#include "population/include/Population.hpp"
#include "population/include/Brain.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <ctime>

// Definições dos globais de domain randomization (declarados extern em Agent.hpp).
// No treino headless deixamos 0 (treino limpo).
double g_noise_std    = 0.0;
int    g_delay_frames = 0;

static bool saveBrain(const Brain& b, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(17);
    f << b.numInput << " " << b.numHidden << "\n";
    for (double x : b.weights_hidden) f << x << "\n";
    for (double x : b.biases_hidden)  f << x << "\n";
    for (double x : b.weights_output) f << x << "\n";
    f << b.biases_output << "\n";
    return (bool)f;
}

// Finaliza o fitness de pássaros vivos (mesma regra do Agent::act na morte).
static void finalizeAlive(Population& pop) {
    for (auto& a : pop.agents) {
        if (a.bird.alive) {
            a.fitness = (a.bird.score == 0)
                        ? (float)(a.bird.time * a.bird.time)
                        : 50.0f * a.bird.score;
        }
    }
}

int main(int argc, char** argv) {
    int generations    = (argc > 1) ? std::atoi(argv[1]) : 300;
    std::string model  = (argc > 2) ? argv[2] : "best_brain.txt";

    std::srand((unsigned)std::time(nullptr));

    const int initialBirds = 1000;
    const double dt = 0.016;
    const int STEP_CAP = 20000;   // ~5min de jogo por geração: corta birds imortais

    Population generation(initialBirds);
    World physics(initialBirds);

    int elite = 5, parents = 10, tournamentSize = 10;

    float birdX = generation.agents[0].bird.position.x;
    float birdRadius = generation.agents[0].bird.radius;

    int bestEver = -1;

    for (int g = 0; g < generations; g++) {
        int steps = 0;
        while (physics.birdsAlive > 0 && steps < STEP_CAP) {
            physics.updateObstacle(dt);
            if (physics.updateScoreEval(birdX - birdRadius))
                generation.updateScore();

            for (int i = 0; i < initialBirds; i++) {
                Vector2 nextObstacle = physics.getInputs();
                physics.updateBird(dt, generation.agents[i].bird);
                physics.handleCollision(generation.agents[i].bird);
                generation.agents[i].act(nextObstacle.x, nextObstacle.y, dt);
            }
            steps++;
        }

        finalizeAlive(generation);   // caso tenha batido o STEP_CAP com vivos
        int idx = elite;
        generation.sort();

        int score = physics.maximumScore;
        if (score > bestEver) bestEver = score;
        std::cout << "Gen " << (g + 1) << "/" << generations
                  << "  score=" << score
                  << "  best_fitness=" << generation.agents[0].fitness
                  << "  (recorde=" << bestEver << ")\n";

        saveBrain(generation.agents[0].brain, model);

        generation.getEliteChildren(idx, parents);
        generation.getDiverseChildren(idx, tournamentSize);
        generation.restartBirds();
        generation.gen++;
        physics.reset(initialBirds);
    }

    std::cout << "[OK] Treino concluido. Melhor score visto=" << bestEver
              << ". Modelo salvo em " << model << "\n";
    return 0;
}
