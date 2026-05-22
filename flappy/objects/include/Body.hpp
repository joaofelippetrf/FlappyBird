#pragma once
#include <SFML/Graphics.hpp>
#include "Vector2.hpp"

struct Body {
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    float radius;
    sf::CircleShape shape; 
    bool alive;
    int score =0;
    float time = 0;
    
    Body(Vector2 pos, Vector2 vel);

    void update(double dt);
    void jump();    
};