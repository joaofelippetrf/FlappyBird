#include <SFML/Graphics.hpp>
#include <optional>
#include <vector>
#include "core/include/World.hpp"
#include "population/include/Agent.hpp"
#include "population/include/Population.hpp"
#include "core/include/Random.hpp"
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <execution>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "population/include/Brain.hpp"

// Rede UDP (POSIX — funciona em Linux, Raspberry Pi e macOS)
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

// Domain randomization (sim-to-real): declarados em Agent.hpp, usados em Agent::act.
double g_noise_std   = 0.0;
int    g_delay_frames = 0;

enum class Mode { Train, Play, Serve };

// Abre um socket UDP não-bloqueante escutando em 0.0.0.0:port.
// Retorna o fd, ou -1 em caso de erro.
static int openJumpSocket(unsigned short port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    return fd;
}

// Drena todos os datagramas pendentes. Retorna true se chegou pelo menos um
// comando de pulo (qualquer datagrama não-vazio conta como "pula").
static bool pollJump(int fd) {
    bool jump = false;
    char buf[64];
    while (true) {
        ssize_t n = recvfrom(fd, buf, sizeof(buf), 0, nullptr, nullptr);
        if (n <= 0) break;   // EWOULDBLOCK => sem mais pacotes
        jump = true;
    }
    return jump;
}

// Persistência simples do cérebro (texto, números em ordem fixa)
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

static bool loadBrain(Brain& out, const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    int ni, nh; f >> ni >> nh;
    if (!f || ni <= 0 || nh <= 0) return false;
    std::vector<double> w_h(ni * nh), b_h(nh), w_o(nh);
    double b_o = 0.0;
    for (auto& x : w_h) f >> x;
    for (auto& x : b_h) f >> x;
    for (auto& x : w_o) f >> x;
    f >> b_o;
    if (!f) return false;
    out = Brain(w_h, b_h, w_o, b_o);
    out.numInput  = ni;
    out.numHidden = nh;
    return true;
}

int main(int argc, char** argv)
{
    Mode mode = Mode::Train;
    std::string model_path = "best_brain.txt";
    unsigned short serve_port = 5005;
    if (argc > 1) {
        std::string m = argv[1];
        if (m == "play")       mode = Mode::Play;
        else if (m == "train") mode = Mode::Train;
        else if (m == "serve") mode = Mode::Serve;
        else {
            std::cerr << "Uso: " << argv[0] << " [train|play|serve] [model_path|porta]\n"
                      << "  train (default): roda evolução com 1000 pássaros, salva o melhor em " << model_path << "\n"
                      << "  play:            carrega 1 pássaro do modelo salvo\n"
                      << "  serve [porta]:   1 pássaro controlado por UDP (cérebro roda em outra máquina, ex.: Raspberry Pi)\n";
            return -1;
        }
    }
    if (mode == Mode::Serve) {
        if (argc > 2) serve_port = static_cast<unsigned short>(std::atoi(argv[2]));
    } else if (argc > 2) {
        model_path = argv[2];
    }
    const char* mode_name = (mode == Mode::Train) ? "TRAIN"
                          : (mode == Mode::Play)  ? "PLAY" : "SERVE";
    std::cout << "[INFO] modo=" << mode_name;
    if (mode == Mode::Serve) std::cout << " porta_udp=" << serve_port;
    else                     std::cout << " model=" << model_path;
    std::cout << "\n";

    // Domain randomization (sim-to-real): treine com TRAIN_NOISE/TRAIN_DELAY
    // pra o cérebro aguentar o ruído e a latência da câmera.
    if (const char* e = std::getenv("TRAIN_NOISE")) g_noise_std    = std::atof(e);
    if (const char* e = std::getenv("TRAIN_DELAY")) g_delay_frames = std::atoi(e);
    if (g_noise_std > 0.0 || g_delay_frames > 0)
        std::cout << "[RANDOMIZE] noise_std=" << g_noise_std
                  << " delay_frames=" << g_delay_frames << "\n";

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    sf::Font font;
    if (!font.openFromFile("FontAwesome6_Regular.ttf"))
    {
        return -1;
    }

    sf::Text uiText(font);
    uiText.setCharacterSize(25);
    uiText.setFillColor(sf::Color::White);
    uiText.setOutlineColor(sf::Color::Black);
    uiText.setOutlineThickness(2.f);
    uiText.setPosition({20.f, 20.f});

    sf::RenderWindow window(sf::VideoMode({1920u, 1080u}), "Flappy bird");
    window.setFramerateLimit(60);
    double dt = 0.016;
    int initialBirds = (mode == Mode::Train) ? 1000 : 1;

    // Modo serve: abre o socket UDP que recebe os pulos da máquina do cérebro.
    int jump_fd = -1;
    if (mode == Mode::Serve) {
        jump_fd = openJumpSocket(serve_port);
        if (jump_fd < 0) {
            std::cerr << "[ERRO] Nao foi possivel abrir socket UDP na porta " << serve_port << "\n";
            return -1;
        }
        std::cout << "[SERVE] Escutando pulos em UDP 0.0.0.0:" << serve_port << "\n";
    }

    Population generation(initialBirds);
    World physics(initialBirds);

    int elite = std::min(5,  initialBirds);
    int parents = std::min(10, initialBirds);
    int tournamentSize = std::min(10, initialBirds);
    int idx ;

    if (mode == Mode::Play) {
        Brain b;
        if (!loadBrain(b, model_path)) {
            std::cerr << "[ERRO] Nao foi possivel carregar modelo de '" << model_path
                      << "'. Rode antes em modo train.\n";
            return -1;
        }
        generation.agents[0].brain = b;
        std::cout << "[INFO] Modelo carregado. w0=" << b.weights_hidden[0]
                  << " (ni=" << b.numInput << " nh=" << b.numHidden << ")\n";
    }

    // CONTINUE=1: treina a partir do best salvo (em vez de começar do zero).
    // Semeia vários agentes com o cérebro carregado pra ele sobreviver à seleção
    // e evoluir; o resto continua aleatório (mantém diversidade).
    if (mode == Mode::Train && std::getenv("CONTINUE")) {
        Brain b;
        if (loadBrain(b, model_path)) {
            int seed = std::min(20, initialBirds);
            for (int i = 0; i < seed; i++) generation.agents[i].brain = b;
            std::cout << "[TRAIN] Continuando de " << model_path
                      << " (" << seed << " sementes).\n";
        } else {
            std::cout << "[TRAIN] CONTINUE pedido mas '" << model_path
                      << "' nao existe — comecando do zero.\n";
        }
    }

    float birdX = generation.agents[0].bird.position.x;
    float birdRadius = generation.agents[0].bird.radius;

    // Ground-truth para validação contra vision.cpp.
    // Coordenadas normalizadas em [0,1] usando o canvas lógico 1920x1080.
    std::ofstream truth("/tmp/flappy_truth.jsonl", std::ios::out | std::ios::trunc);
    const float GAME_W = 1920.f, GAME_H = 1080.f;
    auto wall_seconds = []() {
        auto d = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration<double>(d).count();
    };

    bool forceEndGen = false;   // 'K' encerra a geração atual na hora

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (const auto* key = event->getIf<sf::Event::KeyPressed>())
            {
                // K: mata a geração atual (salva o melhor e evolui já).
                // Esc: fecha (em train salva o melhor da geração atual antes).
                if (key->code == sf::Keyboard::Key::K)      forceEndGen = true;
                else if (key->code == sf::Keyboard::Key::Escape) window.close();
            }
        }
        window.clear(sf::Color::Black);
        physics.updateObstacle(dt);
        physics.draw(window);

        bool updateScore = physics.updateScoreEval(birdX - birdRadius);
        if(updateScore){
            generation.updateScore();
        }

        // No modo serve, o pulo vem por UDP (cérebro roda em outra máquina).
        // Drenamos o socket uma vez por frame, antes de atualizar o pássaro.
        bool remoteJump = false;
        if (mode == Mode::Serve) remoteJump = pollJump(jump_fd);

        for (int i = 0; i < initialBirds; i++)
        {
            Vector2 nextObstacle = physics.getInputs();

            physics.updateBird(dt, generation.agents[i].bird);
            physics.handleCollision(generation.agents[i].bird);
            if (mode == Mode::Serve) {
                // Sem cérebro local: aplica o pulo recebido pela rede.
                if (remoteJump && generation.agents[i].bird.alive)
                    generation.agents[i].bird.jump();
            } else {
                generation.agents[i].act(nextObstacle.x, nextObstacle.y, dt);
            }
            if(generation.agents[i].bird.alive)  physics.drawBird(window, generation.agents[i].bird);
        }

        // HUD (Generation/Alive/Score) desativado: o texto branco no canto
        // virava contorno e atrapalhava a detecção da câmera. O score continua
        // saindo no terminal a cada fim de geração.
        window.display();

        if (truth && physics.obstacles.size() >= 2 && !generation.agents.empty()) {
            const Body& b = generation.agents[0].bird;
            int top_i = (physics.obstacles[0].passed && physics.obstacles.size() >= 4) ? 2 : 0;
            const Obstacle& tp = physics.obstacles[top_i];
            const Obstacle& bp = physics.obstacles[top_i + 1];
            float pipe_x     = (tp.position.x - tp.width * 0.5f) / GAME_W;
            float gap_top    = (tp.position.y + tp.height * 0.5f) / GAME_H;
            float gap_bottom = (bp.position.y - bp.height * 0.5f) / GAME_H;
            truth << std::fixed << std::setprecision(6)
                  << "{\"t\":"          << wall_seconds()
                  << ",\"bird_x\":"     << (b.position.x / GAME_W)
                  << ",\"bird_y\":"     << (b.position.y / GAME_H)
                  << ",\"bird_r\":"     << (b.radius     / GAME_H)
                  << ",\"pipe_x\":"     << pipe_x
                  << ",\"gap_top\":"    << gap_top
                  << ",\"gap_bottom\":" << gap_bottom
                  << ",\"alive\":"      << (b.alive ? 1 : 0)
                  << "}\n";
            truth.flush();
        }

        if (physics.birdsAlive == 0 || forceEndGen)
        {
            if (forceEndGen) {
                std::cout << "[" << mode_name << "] Geracao encerrada manualmente (K).\n";
                forceEndGen = false;
            }
            if (mode == Mode::Play || mode == Mode::Serve) {
                // Play/Serve não evoluem: só recomeçam (o cérebro é externo no serve).
                std::cout << "[" << mode_name << "] Score final: " << physics.maximumScore
                          << " — reiniciando.\n";
                generation.restartBirds();
                physics.reset(initialBirds);
                continue;
            }

            // Pássaros ainda vivos (ex.: fim forçado com K) têm fitness=0 porque
            // o fitness só é calculado na morte. Finaliza aqui pelo score atual,
            // senão o sort joga o melhor (vivo) pro fim e salvamos um cérebro ruim.
            for (auto& a : generation.agents) {
                if (a.bird.alive) {
                    a.fitness = (a.bird.score == 0)
                                ? (float)(a.bird.time * a.bird.time)
                                : 50.0f * a.bird.score;
                }
            }

            idx = elite;
            generation.sort();
            std::cout << "maximumScore at Generation " << generation.gen
                      << ": " << physics.maximumScore << "\n";

            // Salva o melhor cérebro depois do sort (agents[0] = melhor)
            if (!generation.agents.empty()) {
                const Agent& best = generation.agents[0];
                std::cout << "[TRAIN] melhor salvo: score=" << best.bird.score
                          << " fitness=" << best.fitness
                          << " w0=" << best.brain.weights_hidden[0] << "\n";
                if (saveBrain(best.brain, model_path)) {
                    std::cout << "[TRAIN] Modelo salvo em " << model_path << "\n";
                } else {
                    std::cerr << "[TRAIN] Falha ao salvar modelo em " << model_path << "\n";
                }
            }

            generation.getEliteChildren(idx, parents);
            generation.getDiverseChildren(idx, tournamentSize);
            generation.restartBirds();

            generation.gen++;
            physics.reset(initialBirds);
        }
    }

    // Ao sair (Esc/fechar janela) em modo train, salva o melhor da geração
    // atual — assim uma geração em andamento não é perdida. Finaliza o fitness
    // dos vivos antes do sort (mesmo motivo do fim forçado por K).
    if (mode == Mode::Train && !generation.agents.empty()) {
        for (auto& a : generation.agents) {
            if (a.bird.alive) {
                a.fitness = (a.bird.score == 0)
                            ? (float)(a.bird.time * a.bird.time)
                            : 50.0f * a.bird.score;
            }
        }
        generation.sort();
        if (saveBrain(generation.agents[0].brain, model_path))
            std::cout << "[TRAIN] Melhor da geracao atual salvo em " << model_path << " ao sair.\n";
    }

    if (jump_fd >= 0) close(jump_fd);
    return 0;
}