# Halite Challenge with Reinforcement Learning

A cleaned and portfolio-ready version of an academic project based on the **Kaggle Halite** environment. The original work was developed in a notebook as an exploration of **reinforcement learning for resource collection and navigation in a multi-agent grid world**. This repository reorganizes that work into readable modules, adds project documentation, and makes the code easier to present, maintain, and extend.

## Project overview

This project explores how an agent can act in the Halite game environment using two complementary approaches:

- a **rule-based baseline agent** for interpretable behaviour and benchmarking,
- a **simplified actor-critic reinforcement learning prototype** for policy learning.

The main goal is not to present a competition-winning bot, but to demonstrate:

- reinforcement learning workflow design,
- environment interaction and action selection,
- reward shaping and training logic,
- code refactoring from notebook-style research into a reusable repository.

## Why this project matters

This repository is a strong academic-to-portfolio example because it shows:

- transition from theory to implementation,
- handling of game-style sequential decision making,
- reinforcement learning concepts such as **policy**, **value estimation**, and **discounted returns**,
- code cleanup and modernization of an older project.

## Key features

- **Baseline heuristic bot** with collection and deposit rules
- **Actor-critic model** with separate policy and value heads
- **Training loop** with discounted return computation
- **Readable project structure** instead of one long notebook
- **Documentation of outputs, limitations, and modernization path**

## Repository structure

```text
halite_refactor/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── simple_agent.py
│   └── actor_critic_training.py
└── docs/
    ├── project_summary.md
    ├── output_and_description.md
    ├── technical_overview.md
    ├── portfolio_blurb.md
    └── refactor_notes.md
```

## Method summary

### 1. Baseline agent
The baseline agent follows handcrafted game logic:

- create a shipyard if none exists,
- spawn a ship when infrastructure exists but fleet is empty,
- collect halite when cargo is low,
- return to shipyard when cargo is high,
- move toward the richest neighbouring cell when mining locally is inefficient.

This agent is useful as a **reference policy** and helps judge whether a learned policy is actually adding value.

### 2. Reinforcement learning prototype
The RL section uses a simplified **actor-critic architecture**:

- **Input:** flattened Halite board
- **Actor head:** predicts probabilities over actions
- **Critic head:** estimates the value of the current state
- **Reward:** based on score delta rather than repeatedly reusing cumulative score

This version remains intentionally simple to preserve the original academic logic while making the implementation easier to understand.

## Technical stack

- Python
- NumPy
- TensorFlow / Keras
- Kaggle Environments (Halite)
- Matplotlib

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the baseline agent

```bash
python src/simple_agent.py
```

### 3. Run actor-critic training

```bash
python src/actor_critic_training.py
```

## Usage examples

### Example: import the baseline agent

```python
from src.simple_agent import simple_agent
```

### Example: train the RL model

```python
from src.actor_critic_training import TrainingConfig, train_actor_critic

cfg = TrainingConfig(episodes=20, render_after_training=False)
model, running_rewards = train_actor_critic(cfg)
```

### Example: plot the reward curve

```python
import matplotlib.pyplot as plt
from src.actor_critic_training import TrainingConfig, train_actor_critic

cfg = TrainingConfig(episodes=20)
model, running_rewards = train_actor_critic(cfg)

plt.plot(running_rewards)
plt.xlabel("Episode")
plt.ylabel("Running reward")
plt.title("Halite RL training curve")
plt.show()
```

## Outputs

The project produces several outputs:

- **environment actions** from the baseline bot,
- **actor policy probabilities** from the RL model,
- **critic value estimates**,
- **training logs** showing episode-level progress,
- **reward curves** for learning inspection,
- optional **environment render output**.

Detailed descriptions are available in [`docs/output_and_description.md`](docs/output_and_description.md).

## Main limitations

This is still a simplified academic prototype. Important limitations include:

- the state encoder uses only the raw halite map,
- the RL logic mainly acts on the first ship,
- enemy information is not explicitly modeled,
- shipyard strategy remains partly heuristic,
- the project is not yet a full multi-agent competitive system.

These limitations are explained clearly because they are part of the research story and show where future improvement is possible.

## Refactoring improvements made

Compared with the original notebook version, this repository:

- separates code from explanation,
- removes duplicated helper logic,
- makes ship-state handling safer,
- modernizes parts of the TensorFlow usage,
- improves reward interpretation,
- documents architectural constraints and next steps.

See [`docs/refactor_notes.md`](docs/refactor_notes.md) for details.

## Future improvements

Planned extensions that would make this a stronger RL system:

1. richer observation encoding,
2. multi-ship coordinated control,
3. checkpoint saving and loading,
4. evaluation against multiple opponent types,
5. experiment tracking and hyperparameter logging,
6. comparison plots across baseline and RL variants.

## Portfolio summary

This project demonstrates the ability to:

- clean and modernize older academic code,
- explain RL decisions and outputs clearly,
- structure an experimental ML project for public presentation,
- convert notebook-based work into a reusable repository.

A short portfolio-friendly summary is included in [`docs/portfolio_blurb.md`](docs/portfolio_blurb.md).

## Notes

Depending on the installed version of `kaggle-environments`, small interface adjustments may be required.

