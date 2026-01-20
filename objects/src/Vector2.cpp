#include "../include/Vector2.hpp"
#include <cmath>

Vector2 Vector2::operator+(const Vector2& other) const {
    return { x + other.x, y + other.y };
};

Vector2 Vector2::operator-(const Vector2& other) const {
    return { x - other.x, y - other.y };
}

Vector2 Vector2::operator*(double scale) const {
    return { x * scale, y * scale };
}

double Vector2::magnitudeSq() const {
    return { x * x + y * y };
}

Vector2 Vector2::operator/(double scale) const {
    if (std::abs(scale) < 1e-9) return { 0, 0 }; 
    return { x / scale, y / scale };
}