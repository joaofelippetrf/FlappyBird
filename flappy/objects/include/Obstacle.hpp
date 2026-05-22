#pragma once
#include <SFML/Graphics.hpp>
#include "Vector2.hpp"

struct Obstacle {
    Vector2 position;
    Vector2 velocity;
    float width;
    float height;
    sf::RectangleShape shape;
    bool passed = false;
    Obstacle(Vector2 pos,  float height);
    
    void update(double dt);
};