#include <SFML/Graphics.hpp>
#include <optional>
#include <vector>
#include "core/include/World.hpp"
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <execution>

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    sf::RenderWindow window(sf::VideoMode({1920u,  1080u}), "Flappy bird");
    window.setFramerateLimit(60);
    World physics ;
    double dt = 0.016;
    Body bird = physics.birde;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>() ) {
                window.close();
            }
            physics.handleInput(*event, bird);
        }
        
        if(physics.game == false) return 0;
        window.clear(sf::Color::Black);
        physics.handleCollision(bird);
        physics.update(dt, bird);
        physics.draw(window, bird);
        window.display();
    }

    return 0;
}

// em world tornar bird um vector
// em main passar (brain, bird, obstacles) para o agente
// em main que se cria uma nova geração, desse jeito apenas brain acessa world
