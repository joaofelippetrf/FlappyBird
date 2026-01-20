#include <vector>
#include <SFML/Graphics.hpp>
#include "../../objects/include/Body.hpp"
#include "../../objects/include/Obstacle.hpp"
#include "../include/World.hpp"
#include <iostream>


World::World() 
    : birde({200, 540}, {0,0}), game(true), score(0)
{ 
}

void World :: handleInput(const sf::Event& event, Body& bird){        
        if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {        
            if (keyPressed->code == sf::Keyboard::Key::Up ) {
                bird.jump(); 
            }
        }
        if (event.is<sf::Event::MouseButtonPressed>()) {
            bird.jump();
        }
}

void World :: handleCollision(Body& bird){        
    if(obstacles.empty()) return;
    float halfWidth = obstacles[0].width/2.0;
    float bX = bird.position.x;
    float bY = bird.position.y;
    float rad = bird.radius;

    if(bY >= 1080 || bY <= -200) {game = false; return;}
        
    float objX = obstacles[0].position.x ;

    if(objX - halfWidth - rad > bX) return;
    if(objX + halfWidth + rad < bX) return;

    float top = obstacles[0].position.y + obstacles[0].height /2.0;
    float bottom = obstacles[1].position.y - obstacles[1].height /2.0;

    if(abs(bX - objX) <= halfWidth){
        if(bY - rad < top || bY + rad >bottom ) game = false;
        if(abs(bY - top) >= rad && abs(bY - bottom) >= rad ) return;
        game = false;
        return;
    }

    if(objX <bX) objX += halfWidth;
    else objX -= halfWidth;

    double distance1 = (Vector2{objX, top} - bird.position). magnitudeSq();
    double distance2 = (Vector2{objX, bottom} - bird.position). magnitudeSq();
    double rdSq = (double)rad * rad;

    if(distance1 > rdSq && distance2 > rdSq) return;
    game = false;
    return;
}

void World :: update(double dt, Body& bird){
    float timer =1.4;
    if(spawn >= timer || obstacles.empty()){
        float screenHeight = 1080.f;
        float empty = 250.f; 

        int rdm = (rand() % 580) + 200;


        float heightUp = rdm - (empty / 2.f); 
        Obstacle up(Vector2{1920, heightUp / 2.f}, heightUp);

        float heightDown = screenHeight - (rdm + (empty / 2.f));
        Obstacle down(Vector2{1920, screenHeight - (heightDown / 2.f)}, heightDown);

        std :: vector<Obstacle> nw;
        
        for(int i = 0;i < obstacles.size() ; i++){
            if(obstacles[i].position.x <=0){ 
                i++;
                continue;
            }
            else {
                obstacles[i].update(dt);
                nw.emplace_back(obstacles[i]);
            }
        }

        obstacles = nw;
        obstacles.emplace_back(up);
        spawn = 0;
        obstacles.emplace_back(down);
    }

    else{
        for(auto& obs: obstacles){
            obs.update(dt);
        }
    }
    
    bool hasPassed = bird.position.x - bird.radius > obstacles[0].position.x + obstacles[0].width/2.0;
    bird.update(dt);
    if(!hasPassed && bird.position.x - bird.radius > obstacles[0].position.x + obstacles[0].width/2.0) score++; 
    spawn+=dt;
}

void World :: draw(sf::RenderWindow& window, Body& bird){
    for(auto& obstacle : obstacles){
        window.draw(obstacle.shape);
    }
    window.draw(bird.shape);
}