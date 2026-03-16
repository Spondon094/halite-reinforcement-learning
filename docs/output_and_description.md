# Output and Description

This document explains the main outputs from the cleaned Halite project and what each output means.

## 1. Baseline agent output

**Produced by:** `src/simple_agent.py`

### What it returns
The function `simple_agent(obs, config)` returns a dictionary of Halite actions for the current player.

### What those actions mean
- `SPAWN`: Create a ship at a shipyard.
- `CONVERT`: Convert a ship into a shipyard.
- `NORTH`, `SOUTH`, `EAST`, `WEST`: Move the ship one tile.

### Behaviour summary
The baseline agent uses cargo thresholds:
- below `200` halite -> collect,
- above `500` halite -> deposit.

When the ship is in collect mode and its current tile has low halite, it moves toward the richest adjacent tile.  
When it is in deposit mode, it moves back toward the first shipyard.

### Practical interpretation
This output is not a learned policy.  
It is a deterministic heuristic baseline that helps you compare whether RL is actually improving behaviour.

---

## 2. Model summary output

**Produced by:** `build_actor_critic()` in `src/actor_critic_training.py`

### What it represents
The actor-critic model has two heads:
- **Actor head**: predicts probabilities over 5 actions.
- **Critic head**: predicts the value of the current state.

### Input
A flattened 21 x 21 halite board:
- input dimension = `441`

### Actor output
A probability vector of length `5`:
1. `NORTH`
2. `SOUTH`
3. `EAST`
4. `WEST`
5. `CONVERT`

### Critic output
A single scalar value estimating the long-term usefulness of the current state.

### Practical interpretation
If the actor assigns high probability to one action, it means the policy currently prefers that action in the observed state.  
If the critic value is high, it means the model expects that state to lead to better future reward.

---

## 3. Training log output

**Produced by:** `train_actor_critic()`

Example console message:
```text
Episode 005 | running_reward=0.0132
```

### What it means
- `Episode 005`: the training episode number
- `running_reward`: exponentially smoothed episode reward

### Why it matters
This is your main high-level indicator during training.
- If it trends upward, the policy may be improving.
- If it is flat or unstable, the representation, reward, or policy logic may be too weak.

---

## 4. Reward curve output

**Produced by:** the `running_rewards` list returned by `train_actor_critic()`

### What it contains
A list of smoothed rewards, one value per episode.

### How to use it
You can plot it with Matplotlib:

```python
import matplotlib.pyplot as plt

plt.plot(running_rewards)
plt.xlabel("Episode")
plt.ylabel("Running reward")
plt.title("Halite Actor-Critic Training Curve")
plt.show()
```

### Practical interpretation
- upward curve -> training signal may be useful,
- noisy curve -> unstable learning,
- flat near zero -> little measurable progress.

---

## 5. Environment render output

**Produced by:** `env.render(mode="ipython", width=800, height=600)`

### What it shows
A visual replay of the Halite game environment.

### Why it is important
Numeric reward alone is not enough in game AI.
You should also inspect whether the bot:
- mines effectively,
- returns cargo safely,
- avoids wasteful movement,
- converts and spawns at sensible moments.

---

## 6. Reward definition used in the cleaned script

The original notebook used the player's cumulative bank repeatedly inside the episode:

```python
gain = state.players[0][0] / 5000
```

That approach can over-count reward because the same accumulated score keeps being reused at many timesteps.

### Cleaner replacement used in the refactor
The cleaned script uses **score delta**:

```python
reward = (current_bank - previous_bank) / reward_scale
```

### Why this is better
This better matches what changed at the current step, so the training signal is easier to interpret.

---

## 7. Important limitation of the current RL setup

Even after cleanup, this project is still a simplified academic prototype.

### Main limitations
- Only the raw halite map is used as input.
- The policy effectively controls only the first ship.
- Enemy information is not encoded in the state.
- Shipyard placement is still mostly heuristic.
- This is not a full competition-ready multi-agent system.

### What that means for the output
The results are useful as:
- an educational RL project,
- a portfolio artifact,
- a starting point for improvement,

but not yet as a strong Halite competition bot.
