# Technical Overview

## Environment
The project uses the Kaggle Halite environment, a turn-based grid world in which agents collect halite, move strategically, and return resources to convert them into score.

## Baseline policy
The baseline agent is rule-driven and uses simple threshold logic:

- build a shipyard if none exists,
- spawn ships when infrastructure is available,
- collect when cargo is low,
- deposit when cargo is high,
- choose locally profitable movement when mining should continue.

This policy provides a transparent benchmark for comparison.

## RL model
The reinforcement learning model follows an actor-critic design:

- the **actor** outputs action probabilities,
- the **critic** estimates state value,
- both heads share the same environment-facing state input.

## State representation
The current state representation is intentionally simple: the halite board is flattened into a one-dimensional vector. This makes the project easy to inspect, but it also limits strategic awareness.

## Reward design
The refactored version uses **bank score delta** as the reward signal. This is easier to interpret than repeatedly using cumulative score during an episode and provides a clearer relationship between action consequences and learning signal.

## Training loop
For each episode:

1. the environment is reset,
2. the model predicts an action distribution,
3. an action is sampled,
4. the environment is stepped forward,
5. rewards are recorded,
6. discounted returns are computed,
7. actor and critic losses are combined,
8. gradients are applied with Adam.

## Current constraints
This implementation remains a simplified research prototype:

- limited state encoding,
- first-ship control bias,
- hybrid policy logic between learned and heuristic decisions,
- no checkpointing or formal evaluation framework yet.

## Recommended next upgrade path
- add richer features for ships, cargo, shipyards, and enemies,
- support multi-ship action generation,
- add reproducible experiment configuration,
- compare baseline and RL agents quantitatively,
- save training artifacts and plots automatically.

