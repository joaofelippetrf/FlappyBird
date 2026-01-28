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

int main()
{
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
    int initialBirds = 1000;
    
    Population generation(initialBirds);
    World physics(initialBirds);

    int elite =5;
    int parents = 10;
    int tournamentSize = 10;
    int idx ;

    float birdX = generation.agents[0].bird.position.x;
    float birdRadius = generation.agents[0].bird.radius;

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }
        window.clear(sf::Color::Black);
        physics.updateObstacle(dt);
        physics.draw(window);

        bool updateScore = physics.updateScoreEval(birdX - birdRadius);
        if(updateScore){
            generation.updateScore();
        }

        for (int i = 0; i < initialBirds; i++)
        {
            Vector2 nextObstacle = physics.getInputs();
            
            physics.updateBird(dt, generation.agents[i].bird);
            physics.handleCollision(generation.agents[i].bird);
            generation.agents[i].act(nextObstacle.x, nextObstacle.y, dt);
            if(generation.agents[i].bird.alive)  physics.drawBird(window, generation.agents[i].bird);
        }

        std::string stats = "Generation " + std::to_string(generation.gen) +
                            "\nAlive " + std::to_string(physics.birdsAlive) +
                            "\nScore " + std::to_string(physics.maximumScore);
        uiText.setString(stats);
        window.draw(uiText);

        window.display();

        if (physics.birdsAlive ==0)
        {
            if (physics.birdsAlive != 0 && generation.gen >= 20) continue;
            idx = elite;
            
            generation.sort();
            std::cout << "maximumScore at Generation " <<generation.gen<<": "<< physics.maximumScore<< "\n";
            generation.getEliteChildren(idx, parents);
            generation.getDiverseChildren(idx, tournamentSize);
            generation.restartBirds();

            generation.gen++;
            physics.reset(initialBirds);
        }
    }
    return 0;
}