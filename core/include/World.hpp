#pragma once
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../objects/include/Body.hpp"
#include "../../objects/include/Obstacle.hpp"

struct World {
    Body birde;
    std:: vector<Obstacle> obstacles;
    float spawn =0;
    bool game;
    int score ;
    World(); 

    void handleInput(const sf::Event& event, Body& bird);
    void update(double dt, Body& bird);
    void handleCollision(Body& bird);
    void draw(sf::RenderWindow& window, Body& bird);
};