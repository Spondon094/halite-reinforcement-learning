# 🎮 Halite Challenge with Reinforcement Learning

A portfolio-ready refactor of an academic project built on the **Kaggle Halite** environment. The original work was a notebook exploring **reinforcement learning for resource collection and navigation in a multi-agent grid world**. This repository reorganises that work into clean, readable modules with proper documentation — easier to understand, present, and build on.

---

## 📌 Project overview

The project approaches the Halite game environment through two complementary strategies:

- a **rule-based baseline agent** for interpretable behaviour and benchmarking,
- a **simplified actor-critic RL prototype** for policy learning.

The goal isn't a competition-winning bot. It's a demonstration of:

- reinforcement learning workflow design,
- environment interaction and action selection,
- reward shaping and training logic,
- notebook-to-repository refactoring in practice.

---

## 💡 Why this project matters

This is a clear example of taking academic work seriously enough to clean it up. It shows:

- transition from theory to working implementation,
- handling sequential decision making in a game environment,
- core RL concepts — **policy**, **value estimation**, and **discounted returns** — applied concretely,
- what it looks like to modernise and document older research code.

---

## ✨ Key features

- **Baseline heuristic bot** with collection and deposit rules
- **Actor-critic model** with separate policy and value heads
- **Training loop** with discounted return computation
- **Readable project structure** instead of one long notebook
- **Documentation of outputs, limitations, and a modernisation path**

---

## 🗂️ Repository structure

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

---

## 🧠 Method summary

### 1. Baseline agent

The baseline agent follows handcrafted game logic:

- create a shipyard if none exists,
- spawn a ship when infrastructure exists but the fleet is empty,
- collect halite when cargo is low,
- return to the shipyard when cargo is high,
- move toward the richest neighbouring cell when local mining is inefficient.

This agent serves as a **reference policy** — a way to judge whether a learned policy is actually adding value.

### 2. Reinforcement learning prototype

The RL section uses a simplified **actor-critic architecture**:

- **Input:** flattened Halite board
- **Actor head:** predicts probabilities over actions
- **Critic head:** estimates the value of the current state
- **Reward:** based on score delta rather than cumulative score reuse

This version is intentionally simple to preserve the original academic logic while keeping the implementation easy to follow.

---

## 🛠️ Technical stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 Quick start

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

---

## 💻 Usage examples

### Import the baseline agent

```python
from src.simple_agent import simple_agent
```

### Train the RL model

```python
from src.actor_critic_training import TrainingConfig, train_actor_critic

cfg = TrainingConfig(episodes=20, render_after_training=False)
model, running_rewards = train_actor_critic(cfg)
```

### Plot the reward curve

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

---

## 📊 Outputs

Training and evaluation produce:

- **environment actions** from the baseline bot,
- **actor policy probabilities** from the RL model,
- **critic value estimates**,
- **training logs** showing episode-level progress,
- **reward curves** for inspecting learning behaviour,
- optional **environment render output**.

Full descriptions are in [`docs/output_and_description.md`](docs/output_and_description.md).

---

## ⚠️ Limitations

This is a simplified academic prototype. Known constraints include:

- the state encoder uses only the raw halite map,
- the RL logic mainly acts on the first ship,
- enemy information is not explicitly modelled,
- shipyard strategy remains partly heuristic,
- it is not yet a full multi-agent competitive system.

These limitations are documented deliberately — they are part of the research story and point clearly to where the work could go next.

---

## 🔧 Refactoring improvements

Compared with the original notebook, this repository:

- separates code from explanation,
- removes duplicated helper logic,
- makes ship-state handling safer,
- modernises parts of the TensorFlow usage,
- improves reward interpretation,
- documents architectural constraints and next steps.

Full notes in [`docs/refactor_notes.md`](docs/refactor_notes.md).

---

## 🔭 Future improvements

Extensions that would make this a stronger RL system:

1. richer observation encoding,
2. multi-ship coordinated control,
3. checkpoint saving and loading,
4. evaluation against multiple opponent types,
5. experiment tracking and hyperparameter logging,
6. comparison plots across baseline and RL variants.

---

## 📁 Portfolio summary

This project demonstrates the ability to:

- clean and modernise older academic code,
- explain RL decisions and outputs clearly,
- structure an experimental ML project for public presentation,
- convert notebook-based work into a reusable repository.

A short portfolio-friendly summary is in [`docs/portfolio_blurb.md`](docs/portfolio_blurb.md).

---

## 📝 Notes

Depending on the installed version of `kaggle-environments`, small interface adjustments may be required.
