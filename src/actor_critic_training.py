"""Cleaned Actor-Critic training script derived from the original Halite notebook.

Important notes
---------------
This refactor keeps the academic spirit of the original notebook but makes the code
modular, readable, and easier to extend. It is still a simplified research prototype:

- It uses the flattened halite map as the input state.
- The policy only chooses from five high-level actions:
  NORTH, SOUTH, EAST, WEST, CONVERT
- The control logic applies those actions to the first ship, while still using
  simple heuristics for collection and deposit behaviour.
- It is suitable as a cleaned educational baseline, not as a competition-grade bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction

ACTION_LABELS = ["NORTH", "SOUTH", "EAST", "WEST", "CONVERT"]
ACTION_TO_DIRECTION = {
    "NORTH": ShipAction.NORTH,
    "SOUTH": ShipAction.SOUTH,
    "EAST": ShipAction.EAST,
    "WEST": ShipAction.WEST,
}


@dataclass
class TrainingConfig:
    """Configuration for training."""

    episodes: int = 50
    gamma: float = 0.99
    learning_rate: float = 7e-4
    hidden_units: Tuple[int, int] = (128, 32)
    max_steps_per_episode: int = 500
    reward_scale: float = 5000.0
    render_after_training: bool = False
    random_seed: int = 123


@dataclass
class AgentMemory:
    """Persistent agent state across environment steps."""

    ship_states: Dict[str, str] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """Set NumPy and TensorFlow seeds for partial reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_direction_to(from_pos, to_pos, size: int) -> Optional[ShipAction]:
    """Return a greedy direction from one board position to another."""
    from_x, from_y = divmod(from_pos[0], size), divmod(from_pos[1], size)
    to_x, to_y = divmod(to_pos[0], size), divmod(to_pos[1], size)

    if from_y < to_y:
        return ShipAction.NORTH
    if from_y > to_y:
        return ShipAction.SOUTH
    if from_x < to_x:
        return ShipAction.EAST
    if from_x > to_x:
        return ShipAction.WEST
    return None


def build_actor_critic(
    input_dim: int = 441,
    num_actions: int = 5,
    hidden_units: Sequence[int] = (128, 32),
) -> tf.keras.Model:
    """Build a small actor-critic network."""
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="halite_state")
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="tanh")(x)

    actor_output = tf.keras.layers.Dense(
        num_actions, activation="softmax", name="actor_policy"
    )(x)

    critic_hidden = tf.keras.layers.Dense(hidden_units[0])(inputs)
    critic_hidden = tf.keras.layers.ReLU()(critic_hidden)
    critic_hidden = tf.keras.layers.Dense(hidden_units[1])(critic_hidden)
    critic_hidden = tf.keras.layers.ReLU()(critic_hidden)
    critic_output = tf.keras.layers.Dense(1, name="critic_value")(critic_hidden)

    return tf.keras.Model(inputs=inputs, outputs=[actor_output, critic_output])


def encode_state(state) -> np.ndarray:
    """Flatten the halite map into a 1D float vector."""
    return np.asarray(state.halite, dtype=np.float32).reshape(-1)


def choose_environment_action(obs, config, action_index: int, memory: AgentMemory):
    """Translate the network action into Halite commands.

    The policy only controls the first ship. This mirrors the original notebook,
    but the code below makes the decision logic explicit and safer.
    """
    board = Board(obs, config)
    me = board.current_player
    chosen_label = ACTION_LABELS[action_index]

    # If the fleet is empty but a shipyard exists, try to respawn.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
        return me.next_actions

    # If no shipyard exists, convert the first available ship.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT
        return me.next_actions

    if len(me.ships) == 0:
        return me.next_actions

    ship = me.ships[0]
    memory.ship_states.setdefault(ship.id, "COLLECT")

    # Original threshold-style mode switching.
    if ship.halite < 200:
        memory.ship_states[ship.id] = "COLLECT"
    elif ship.halite > 800:
        memory.ship_states[ship.id] = "DEPOSIT"

    # Allow explicit conversion only if the policy picked it.
    if chosen_label == "CONVERT":
        ship.next_action = ShipAction.CONVERT
        return me.next_actions

    if memory.ship_states[ship.id] == "COLLECT":
        # Only move if the current square is not attractive enough.
        if ship.cell.halite < 100:
            ship.next_action = ACTION_TO_DIRECTION[chosen_label]

    elif memory.ship_states[ship.id] == "DEPOSIT" and len(me.shipyards) > 0:
        direction = get_direction_to(ship.position, me.shipyards[0].position, config.size)
        if direction is not None:
            ship.next_action = direction

    return me.next_actions


def compute_discounted_returns(
    rewards: Sequence[float], gamma: float, eps: float = 1e-8
) -> np.ndarray:
    """Compute normalized discounted returns."""
    returns: List[float] = []
    discounted_sum = 0.0

    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    returns_array = np.asarray(returns, dtype=np.float32)
    returns_array = (returns_array - returns_array.mean()) / (returns_array.std() + eps)
    return returns_array


def train_actor_critic(config: TrainingConfig):
    """Train the actor-critic agent against a random opponent.

    Returns
    -------
    model : tf.keras.Model
        Trained actor-critic network.
    running_rewards : list[float]
        Smoothed reward curve across episodes.
    """
    set_seed(config.random_seed)

    model = build_actor_critic(hidden_units=config.hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    huber_loss = tf.keras.losses.Huber()

    env = make("halite", debug=True)
    trainer = env.train([None, "random"])

    running_reward = 0.0
    running_rewards: List[float] = []

    for episode in range(config.episodes):
        state = trainer.reset()
        memory = AgentMemory()

        action_log_probs: List[tf.Tensor] = []
        critic_values: List[tf.Tensor] = []
        rewards: List[float] = []
        episode_reward = 0.0

        with tf.GradientTape() as tape:
            for _ in range(config.max_steps_per_episode):
                state_vector = encode_state(state)
                state_tensor = tf.convert_to_tensor(state_vector[None, :], dtype=tf.float32)

                action_probs, critic_value = model(state_tensor, training=True)
                critic_values.append(critic_value[0, 0])

                sampled_action = np.random.choice(
                    len(ACTION_LABELS), p=np.squeeze(action_probs.numpy())
                )
                action_log_probs.append(tf.math.log(action_probs[0, sampled_action] + 1e-8))

                previous_bank = state.players[0][0]
                halite_commands = choose_environment_action(
                    state, env.configuration, sampled_action, memory
                )

                # Kaggle trainer returns a state object in the first position.
                state = trainer.step(halite_commands)[0]
                current_bank = state.players[0][0]

                # Cleaner reward shaping than the notebook:
                # use the bank delta rather than repeatedly reusing cumulative score.
                reward = (current_bank - previous_bank) / config.reward_scale
                rewards.append(float(reward))
                episode_reward += float(reward)

                if env.done:
                    break

            returns = compute_discounted_returns(rewards, config.gamma)

            actor_losses: List[tf.Tensor] = []
            critic_losses: List[tf.Tensor] = []

            for log_prob, value, ret in zip(action_log_probs, critic_values, returns):
                advantage = ret - value
                actor_losses.append(-log_prob * advantage)
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            total_loss = tf.reduce_sum(actor_losses) + tf.reduce_sum(critic_losses)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_reward = 0.05 * episode_reward + (1.0 - 0.05) * running_reward
        running_rewards.append(float(running_reward))

        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1:03d} | running_reward={running_reward:.4f}")

    if config.render_after_training:
        env.render(mode="ipython", width=800, height=600)

    return model, running_rewards


if __name__ == "__main__":
    cfg = TrainingConfig(episodes=50, render_after_training=False)
    model, rewards = train_actor_critic(cfg)
    print("Training finished.")
    print(f"Collected {len(rewards)} running-reward values.")
