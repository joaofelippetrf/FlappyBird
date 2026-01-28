#include "../include/Vector2.hpp"
#include "../include/Body.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>

Body::Body(Vector2 pos, Vector2 vel) 
    : position(pos), velocity(vel), radius(30), alive(true)
{
    acceleration = {0, 1200};
    shape.setRadius(radius);
    shape.setFillColor(sf :: Color(255,255,255));

    shape.setOrigin({radius,radius}); 
    shape.setPosition({ static_cast<float>(position.x), static_cast<float>(position.y) });
}

void Body::update(double dt) {
    this->position = this->position + (this->velocity * dt) + this->acceleration*(0.5*dt*dt);

    this->velocity = this->velocity + this->acceleration * dt;

    shape.setPosition({ static_cast<float>(position.x), static_cast<float>(position.y) });
}

void Body::jump() {
    if(this->velocity.y<= 0) this->velocity.y -= 350;
    else this->velocity.y -= 500;
}