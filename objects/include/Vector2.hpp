#pragma once

struct Vector2 {
    double x, y; 

    Vector2 operator+(const Vector2& other) const;
    Vector2 operator-(const Vector2& other) const;
    Vector2 operator*(double scale) const;
    Vector2 operator/(double scale) const;
    double magnitudeSq() const;
};