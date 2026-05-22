#pragma once
#include <vector>
#include "../../objects/include/Body.hpp"
#include "../../objects/include/Obstacle.hpp"

struct World {
    std:: vector<Obstacle> obstacles;
    float spawn =0;
    int birdsAlive;
    int hitBounderings =0;
    int gen =1;
    int obstacleIndex =0;
    int maximumScore =0;
    World(int initialBirds); 

    
    Vector2 getInputs();
    void updateBird(double dt, Body& bird);
    bool updateScoreEval(float leftmostOfBird);
    void updateObstacle(double dt);
    void handleCollision(Body& bird);
    void drawBird(sf::RenderWindow& window, Body& bird);
    void draw(sf::RenderWindow& window);
    void reset(int initialBirds);
};