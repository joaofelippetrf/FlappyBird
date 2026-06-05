#include "../../objects/include/Body.hpp"
#include "../../objects/include/Obstacle.hpp"
#include "../include/World.hpp"
#include <vector>
#include <random>



World::World(int initialBirds) : birdsAlive(initialBirds)
{   
}

Vector2 World :: getInputs(){   
    float x, y;     
    if (obstacles[0].passed) {
        x = obstacles[2].position.x;
        y = obstacles[2].position.y + obstacles[2].height / 2;
    }
    else{
        x = obstacles[0].position.x;
        y = obstacles[0].position.y + obstacles[0].height / 2;
    }
    return {x ,y};
}

void World :: handleCollision(Body& bird){    
    if(obstacles.empty() || !bird.alive) return;

    float halfWidth = obstacles[0].width/2.0;
    float bX = bird.position.x;
    float bY = bird.position.y;
    float rad = bird.radius;

    if(bY >= 1080 || bY <= -100) {bird.alive = false; birdsAlive-- ; hitBounderings++;return;}
        
    float objX = obstacles[0].position.x ;

    if(objX - halfWidth - rad > bX) return;
    if(objX + halfWidth + rad < bX) return;

    float top = obstacles[0].position.y + obstacles[0].height /2.0;
    float bottom = obstacles[1].position.y - obstacles[1].height /2.0;

    if(abs(bX - objX) <= halfWidth){
        if(bY - rad < top || bY + rad >bottom ){ bird.alive = false; birdsAlive--; return;}
        if(abs(bY - top) >= rad && abs(bY - bottom) >= rad ) return;
        bird.alive = false;
        birdsAlive--;
        return;
    }

    if(objX <bX) objX += halfWidth;
    else objX -= halfWidth;

    double distance1 = (Vector2{objX, top} - bird.position). magnitudeSq();
    double distance2 = (Vector2{objX, bottom} - bird.position). magnitudeSq();
    double rdSq = (double)rad * rad;

    if(distance1 > rdSq && distance2 > rdSq) return;
    bird.alive = false;
    birdsAlive--;
    return;
}

void World :: updateBird(double dt, Body& bird){
    if(!bird.alive) return;
    bird.time += dt;
    bird.update(dt);
    if(bird.score > maximumScore) maximumScore = bird.score;

}

bool World :: updateScoreEval(float leftmostOfBird){
        if (obstacles[0].passed == false && leftmostOfBird > obstacles[0].position.x + obstacles[0].width / 2.0){
            obstacles[0].passed = true;
            return true;    
        }
        return false;
}

void World :: updateObstacle(double dt){
    float timer = 2.2;   // mais tempo entre canos (era 1.4) — compensa a velocidade menor

    if(spawn >= timer || obstacles.empty()){
        float screenHeight = 1080.f;
        float empty = 400.f;     // gap maior (era 250) — mais fácil de passar
        int   minPipe = 200;     // altura mínima de CADA cano — garante cima E baixo
                                 // sempre visíveis e grandes o bastante pra câmera.

        // Faixa do centro do gap que garante os dois canos com altura >= minPipe.
        // Com empty=400 e minPipe=200, cada cano fica entre 200 e 480 px.
        int lo = (int)(empty / 2.f) + minPipe;                 // topo do cano de cima >= minPipe
        int hi = (int)(screenHeight - empty / 2.f) - minPipe;  // cano de baixo >= minPipe

        int rdm ;
        if (maximumScore < 40) {
            std::mt19937 deterministicGen(42 + obstacleIndex);
            std::uniform_int_distribution<int> dist(lo, hi);
            rdm = dist(deterministicGen);
            obstacleIndex++;
        }
        else rdm = (rand() % (hi - lo + 1)) + lo;


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
        obstacles.emplace_back(down);
        spawn = 0;
    }

    else{
        for(auto& obs: obstacles){
            obs.update(dt);
        }
    }
    spawn+=dt;
}

void World :: drawBird(sf::RenderWindow& window, Body& bird){
    window.draw(bird.shape);
}

void World :: draw(sf::RenderWindow& window){
    for(auto& obstacle : obstacles){
        window.draw(obstacle.shape);
    }
}

void World :: reset(int initialBirds){
    birdsAlive = initialBirds;
    hitBounderings = 0;
    obstacles.clear();
    obstacleIndex = 0;
    maximumScore = 0;
}