# Workshop 2: Dynamical Systems Analysis & Design

**Authors:**  
J. D. Córdoba Aguirre, A. F. Carvajal Forero  
Universidad Distrital Francisco José de Caldas — May 2025

---

## Overview  

In Workshop 2, we extend the Ms. Pacman AI agent into a full dynamical-systems framework. Building on our previous system design, we:

- Formulate discrete- and continuous-time models of Pacman and ghost motion  
- Define a profit function combining reward and risk  
- Refine cybernetic feedback loops for enhanced perception and adaptive rewards  
- Analyze stability and convergence both theoretically and empirically  
- Design a modular simulation architecture to verify our mathematical model  
- Develop a rigorous testing and evaluation strategy, including phase-portrait analysis  

---

## Key Sections  

### 1. System Dynamics Analysis  

- **Mathematical Model**  
  - Decision-theoretic problem formulation  
  - Continuous- and discrete-time dynamics (ODE → difference equations)  

- **Profit Function**  
  - Reward components for pellets, power-ups, and ghost captures  
  - Risk components via repulsive potentials  

- **Phase Portraits**  
  - Trajectory and state-space plots to reveal attractors and sensitivity  

### 2. Simulation Architecture  

- **Modules**  
  - State Manager, Controller, Dynamics Engine, Reward/Risk Evaluators  
  - Simulation Kernel and Visualization Tools  

- **Implementation**  
  - Python pseudocode for the simulation loop  
  - Functions for updating positions, computing reward, and risk  

### 3. Feedback Loop Refinement  

- **Enhanced Perception**  
  - Add sensors for ghost density, distances, and mode predictions  

- **Granular Rewards**  
  - Context-sensitive incentive adjustments (e.g., higher rewards under threat)  

- **Impact**  
  - Faster learning convergence and more stable policies  

### 4. Stability and Convergence  

- **Stability Criterion**  
  - Bounded trajectories and avoidance of chaotic loops  

- **Convergence Criterion**  
  - Policy variance reduction and flattening of learning curves  

- **Evaluation Methods**  
  - State-norm plots, reward variance tracking, and action-entropy analysis  

---

## Testing and Evaluation Strategy  

- **Controlled Randomness:** Fixed seeds for reproducibility  

- **Scenario Variation:** ROM modifications, map permutations, and ghost-AI tweaks  

- **Performance Metrics:** Cumulative reward, win/loss ratio, episode length, and policy entropy  

- **Stress Testing:** Ghost clustering, dead-end traps, and delayed rewards  

---

## References  

- Paul’s Online Math Notes – *Differential Equations: Phase Plane*  
- UBC Math – *Phase Portraits of Linear Systems*  
- Fabrizio Musacchio – *Using Phase Plane Analysis to Understand Dynamical Systems*  
- Microsoft Research Blog – *Feedback Loops in AI Systems*  
- Analytics Vidhya – *How to Train Ms-Pacman with Reinforcement Learning*  
- AI Stack Exchange – *How Do I Create an AI Controller for Pacman?*
