#include "../include/Vector2.hpp"
#include "../include/Obstacle.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>

Obstacle::Obstacle(Vector2 pos, float height) 
: width(80), height(height)
{
    velocity = {-600,0};
    this->position = pos;

    shape.setSize({width, height});
    shape.setFillColor(sf :: Color (0,255,0));

    shape.setOrigin({width/2,height/2}); 
    shape.setPosition({ static_cast<float>(position.x), static_cast<float>(position.y) });
}

void Obstacle::update(double dt) {
    this->position = this->position + (this->velocity * dt) ;

    shape.setPosition({ static_cast<float>(position.x), static_cast<float>(position.y) });
}
