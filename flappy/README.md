# Flappy Bird AI Agent

An autonomous agent trained to play Flappy Bird using a custom **Neuroevolution** implementation. The project focuses on evolving a neural network's weights through a Genetic Algorithm (GA) to achieve survival and high scores.

## 🧠 Neural Network Architecture

The agent's "brain" consists of a feedforward neural network with a **fixed topology** and a **fully connected graph**.

* **Inputs:** 4 neurons (Environment state)
* **Hidden Layer:** 5 neurons (Single layer)
* **Output Layer:** 1 neuron (Action: Jump or No Jump)

### Input Features
The agent makes decisions based on four spatial and physical variables:
1. **Vertical Velocity**: The bird's current Y-axis speed.
2. **Distance to Ground**: Proximity to the floor.
3. **Horizontal Distance to Center**: X-axis distance to the center of the pipe gap.
4. **Vertical Distance to Center**: Y-axis distance to the center of the pipe gap.

---

## 🧬 Evolution Strategy

The population is optimized using a hybrid Genetic Algorithm designed to balance exploitation of top traits with exploration of new behaviors.

### Selection & Reproduction
* **Tournament Selection:** Used for the majority of the generation to select parents based on fitness.
* **Elitism:** The best-performing agents are preserved directly into the next generation.
* **Elite Procreation:** Top agents breed using **Arithmetic Mean Crossover**, where offspring weights are the average of the parents' weights.
* **Genetic Diversity:** **5%** of every new generation consists of completely random individuals to prevent the population from getting stuck in local optima.

---

## 🛠️ Tech Stack
* **Language:** C++
* **Graphics:** SFML (Simple and Fast Multimedia Library)
* **OS Target:** Windows
* **Build System:** CMake

---

## 🚀 Building and Running

Ensure you have **SFML** and **CMake** installed on your Windows system.

### Build Instructions

1.  **Configure the project:**
    ```bash
    cmake -B build
    ```

2.  **Build the executable:**
    ```bash
    cmake --build build --config Release
    ```
