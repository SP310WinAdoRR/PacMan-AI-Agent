# README.md

## Project Overview

This project models the classic Pacman game as a complex cybernetic system in which the agent (Pacman) is controlled by Deep Reinforcement Learning (DRL). Each system component interacts dynamically, allowing Pacman to learn and adapt in real-time to environmental changes while maximizing point collection and avoiding enemies.

## Functional Specifications

The environment replicates a realistic Pacman maze segmented into quadrants interconnected by corridors and direction-changing nodes. These structures offer opportunities for strategic decisions and dynamic path planning. The agent uses these to navigate the environment, react to threats, and optimize its actions based on reinforcement signals.

## Environment Design

### Map
The map features corridors and nodes that represent junctions for decision-making, which are critical for both Pacman and ghost agents.

### Pellers
Represent positive reinforcement, encouraging Pacman to explore. Power Pellers shift gameplay by enabling Pacman to consume ghosts, creating a temporary reward inversion.

## Agent and Ghost AI

### Pacman (AI-Controlled)
The agent adapts to its environment using DRL, learning over iterations to improve decision-making between exploration and exploitation.

### Ghosts
Operate under three states (Chase, Scatter, Frightened) and adapt based on Pacman’s actions. Each state affects the global strategy of both the ghosts and the agent.

## Stimuli and Rewards

The feedback system integrates positive stimuli (Pellers, Ghost elimination) and negative stimuli (collisions, inactivity), which inform the agent’s reward function. Deep neural networks help process these signals and update Pacman’s policy accordingly.

## Use Cases

1. **Maze Navigation and Peller Collection**: The agent learns optimal trajectories to collect rewards while avoiding high-risk zones.
2. **Power-Up Activation and Ghost Elimination**: Upon activation, the agent switches strategy to capitalize on the temporary advantage of consuming ghosts.
3. **Ghost Avoidance**: Demonstrates the agent’s ability to interpret threats and dynamically plan evasion.

## Feedback Loop System

A complete cybernetic feedback loop guides the agent:

- **Perception**: State input based on environment.
- **Decision**: Policy-based action selection.
- **Action**: Movement executed.
- **Environment Response**: State and reward update.
- **Learning**: Neural network adjustment.

**Three types of loops are defined:**

- **Positive**: Reinforce successful behaviors.
- **Negative**: Discourage harmful or idle actions.
- **Adaptive**: Allow policy change based on ghost behavior.

## Frameworks Used

1. **OpenAI Gym**: For defining the custom grid-based Pacman environment.
2. **OpenAI Retro Gym**: For retro console game simulations using visual input.
3. **PyTorch**: As the primary deep learning engine, offering high flexibility for implementing DQN and reinforcement learning logic.

## References

- OpenAI. (2023). *Gym Documentation*. https://www.gymlibrary.dev/
- OpenAI. (2023). *Retro Documentation*. https://github.com/openai/retro
- PyTorch Team. (2024). *PyTorch Documentation*. https://pytorch.org/docs/stable/index.html
- Gupta, A. (2020). *Feedback Loops in AI Systems*. Microsoft Research. https://www.microsoft.com/en-us/research/blog/feedback-loops-in-ai-systems/
- Corbett, R. (2021). *Understanding Feedback Loops in Reinforcement Learning*. Towards Data Science. https://towardsdatascience.com/understanding-feedback-loops-in-reinforcement-learning-d1bdbf12ddbd

