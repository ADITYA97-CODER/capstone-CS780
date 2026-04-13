# Capstone Project: Reinforcement Learning for Autonomous Exploration

## 📌 Overview

This project focuses on training reinforcement learning agents for autonomous exploration and object-finding tasks. Different architectures and exploration strategies are implemented using Proximal Policy Optimization (PPO), including CNN-based perception, GRU-based memory, and intrinsic motivation via Random Network Distillation (RND).

---

## 📂 Repository Structure

### 1. `ppo_cnn.py`

* Implements **PPO with a Convolutional Neural Network (CNN)**.
* Uses **position memory** to help the agent remember explored areas.
* Designed for spatial awareness using structured observations (e.g., occupancy grids or sensor maps).
* Best suited for learning **exploration with explicit spatial representation**.

---

### 2. `ppo_gru_un.py`

* Implements **PPO with a GRU-based recurrent policy**.
* Includes an **Unwedger mechanism** to help the agent recover from stuck states.
* Enables **temporal memory**, allowing the agent to use past observations and actions.
* Useful for partially observable environments where history matters.

---

### 3. `ppo_ss_rnd.py`

* Implements **Random Network Distillation (RND)** for intrinsic motivation.
* Encourages exploration by rewarding the agent for visiting **novel states**.
* Can be combined with PPO to improve coverage and reduce revisiting behavior.
* Focuses on solving sparse-reward exploration problems.

---

### 4. `submission_cnn.zip`

* Contains:

  * Trained **CNN-based PPO model weights**
  * `agent.py` for evaluation
* Used for evaluating the CNN-based agent in the test environment.

---

### 5. `submission_gru.zip`

* Contains:

  * Trained **GRU-based PPO model weights**
  * `agent.py` for evaluation
* Used for evaluating the GRU-based agent with memory.

---

## ⚙️ Key Features

* PPO-based training across all implementations
* Multiple architectures:

  * CNN (spatial reasoning)
  * GRU (temporal memory)
* Exploration strategies:

  * Position memory
  * RND (intrinsic rewards)
* Modular design for easy experimentation



---

## 🧠 Summary

* `ppo_cnn.py` → PPO + CNN + position memory
* `ppo_gru_un.py` → PPO + GRU + unwedger (memory-based recovery)
* `ppo_ss_rnd.py` → PPO + RND (intrinsic exploration)
* `submission_*.zip` → trained models + evaluation agent

---

## 📌 Notes

* Choose CNN if spatial mapping is reliable.
* Choose GRU if temporal dependencies are critical.
* Use RND when exploration is the main challenge.

---
