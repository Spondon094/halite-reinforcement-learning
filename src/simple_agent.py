"""Baseline rule-based Halite agent extracted and cleaned from the original notebook.

This module preserves the original heuristic idea:
1. Build a shipyard if none exists.
2. Spawn a ship if a shipyard exists and the fleet is empty.
3. Let ships either collect halite or deposit cargo based on cargo thresholds.
4. When collecting, move to the richest neighbouring cell if the current cell is weak.
5. When depositing, move back toward the first shipyard.

The code is intentionally lightweight so it can be used as a baseline for comparison
against more advanced learning-based agents.
"""

from typing import Dict, Optional

from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction

# Cardinal movement directions in a fixed index order.
DIRECTIONS = [
    ShipAction.NORTH,
    ShipAction.EAST,
    ShipAction.SOUTH,
    ShipAction.WEST,
]

# Persistent per-ship mode across turns.
SHIP_STATES: Dict[str, str] = {}


def get_direction_to(from_pos, to_pos, size: int) -> Optional[ShipAction]:
    """Return a greedy direction from one board position to another.

    Parameters
    ----------
    from_pos : Position-like
        Current position of a ship.
    to_pos : Position-like
        Target position (usually a shipyard).
    size : int
        Board size.

    Returns
    -------
    Optional[ShipAction]
        A movement action or None if no move is required.
    """
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


def _best_halite_direction(ship) -> Optional[ShipAction]:
    """Pick the neighbour with the highest halite."""
    neighbours = [
        ship.cell.north.halite,
        ship.cell.east.halite,
        ship.cell.south.halite,
        ship.cell.west.halite,
    ]
    best_index = max(range(len(neighbours)), key=neighbours.__getitem__)
    return DIRECTIONS[best_index]


def simple_agent(obs, config):
    """Rule-based baseline agent for Kaggle Halite."""
    board = Board(obs, config)
    me = board.current_player

    # Recovery rules for empty fleets / empty infrastructure.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    for ship in me.ships:
        if ship.next_action is not None:
            continue

        # Safe default avoids KeyError when a ship is seen for the first time.
        SHIP_STATES.setdefault(ship.id, "COLLECT")

        # Update mode using cargo thresholds from the original notebook.
        if ship.halite < 200:
            SHIP_STATES[ship.id] = "COLLECT"
        elif ship.halite > 500:
            SHIP_STATES[ship.id] = "DEPOSIT"

        # Act based on the current mode.
        if SHIP_STATES[ship.id] == "COLLECT":
            if ship.cell.halite < 100:
                ship.next_action = _best_halite_direction(ship)

        elif SHIP_STATES[ship.id] == "DEPOSIT" and len(me.shipyards) > 0:
            direction = get_direction_to(ship.position, me.shipyards[0].position, config.size)
            if direction is not None:
                ship.next_action = direction

    return me.next_actions
