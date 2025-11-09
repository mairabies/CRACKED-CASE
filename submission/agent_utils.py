"""
State parsing and board utilities for the Case Closed agent.

Provides functions for parsing game state, checking move safety,
and handling torus-wrapped coordinate operations.
"""

from typing import Tuple, List, Optional
from collections import deque

# Board dimensions
HEIGHT = 18
WIDTH = 20

# Cell states
EMPTY = 0
AGENT = 1

# Direction mappings
DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

OPPOSITE_DIR = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT"
}


def parse_state(data: dict) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]], List[Tuple[int, int]], int, int, int, int]:
    """
    Extract game state from JSON data.
    
    Args:
        data: Dictionary with board, trails, boosts, etc.
    
    Returns:
        Tuple of (grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count, player_number)
    """
    grid = data.get("board", [[EMPTY for _ in range(WIDTH)] for _ in range(HEIGHT)])
    agent1_trail = data.get("agent1_trail", [])
    agent2_trail = data.get("agent2_trail", [])
    player_number = data.get("player_number", 1)
    
    if player_number == 1:
        my_trail = agent1_trail
        opp_trail = agent2_trail
        my_boosts = data.get("agent1_boosts", 3)
        opp_boosts = data.get("agent2_boosts", 3)
    else:
        my_trail = agent2_trail
        opp_trail = agent1_trail
        my_boosts = data.get("agent2_boosts", 3)
        opp_boosts = data.get("agent1_boosts", 3)
    
    my_pos = tuple(my_trail[-1]) if my_trail else (0, 0)
    opp_pos = tuple(opp_trail[-1]) if opp_trail else (0, 0)
    turn_count = data.get("turn_count", 0)
    
    return (grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count, player_number)


def torus_wrap(pos: Tuple[int, int]) -> Tuple[int, int]:
    """Normalize coordinates with torus wrapping."""
    x, y = pos
    return (x % WIDTH, y % HEIGHT)


def neighbors(pos: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Get valid adjacent coordinates with torus wrapping.
    
    Args:
        pos: Current position (x, y)
        grid: Board grid
    
    Returns:
        List of valid neighbor positions
    """
    x, y = pos
    candidates = [
        ((x + 1) % WIDTH, y),
        ((x - 1) % WIDTH, y),
        (x, (y + 1) % HEIGHT),
        (x, (y - 1) % HEIGHT)
    ]
    return candidates


def is_safe(cell: Tuple[int, int], grid: List[List[int]], my_trail: List[Tuple[int, int]], opp_trail: List[Tuple[int, int]]) -> bool:
    """
    Check if a cell is safe (not occupied by any trail).
    
    Args:
        cell: Position to check (x, y)
        grid: Board grid
        my_trail: My agent's trail positions
        opp_trail: Opponent's trail positions
    
    Returns:
        True if cell is safe, False otherwise
    """
    wrapped = torus_wrap(cell)
    x, y = wrapped
    
    # Check grid state
    if grid[y][x] == AGENT:
        return False
    
    # Check trails explicitly (redundant but safe)
    if wrapped in my_trail or wrapped in opp_trail:
        return False
    
    return True


def get_current_direction(trail: List[Tuple[int, int]]) -> Optional[str]:
    """
    Infer current direction from trail.
    
    Args:
        trail: List of trail positions
    
    Returns:
        Direction string or None if trail too short
    """
    if len(trail) < 2:
        return None
    
    head = trail[-1]
    prev = trail[-2]
    
    dx = head[0] - prev[0]
    dy = head[1] - prev[1]
    
    # Handle torus wrapping
    if abs(dx) > WIDTH // 2:
        dx = WIDTH - abs(dx) if dx > 0 else -(WIDTH - abs(dx))
    if abs(dy) > HEIGHT // 2:
        dy = HEIGHT - abs(dy) if dy > 0 else -(HEIGHT - abs(dy))
    
    if dx == 1 or dx == -(WIDTH - 1):
        return "RIGHT"
    elif dx == -1 or dx == (WIDTH - 1):
        return "LEFT"
    elif dy == 1 or dy == -(HEIGHT - 1):
        return "DOWN"
    elif dy == -1 or dy == (HEIGHT - 1):
        return "UP"
    
    return None


def legal_moves(grid: List[List[int]], pos: Tuple[int, int], current_dir: Optional[str], my_trail: List[Tuple[int, int]], opp_trail: List[Tuple[int, int]]) -> List[str]:
    """
    Get list of safe legal moves (excluding opposite direction).
    
    Args:
        grid: Board grid
        pos: Current position
        current_dir: Current direction (to exclude opposite)
        my_trail: My trail positions
        opp_trail: Opponent's trail positions
    
    Returns:
        List of safe direction strings
    """
    moves = []
    x, y = pos
    
    for direction, (dx, dy) in DIRECTIONS.items():
        # Skip opposite direction
        if current_dir and direction == OPPOSITE_DIR.get(current_dir):
            continue
        
        new_pos = torus_wrap((x + dx, y + dy))
        if is_safe(new_pos, grid, my_trail, opp_trail):
            moves.append(direction)
    
    return moves


def apply_move(grid: List[List[int]], pos: Tuple[int, int], move: str, my_trail: List[Tuple[int, int]], is_boost: bool = False) -> Tuple[List[List[int]], Tuple[int, int], List[Tuple[int, int]]]:
    """
    Simulate applying a move, returning new state.
    
    Args:
        grid: Current board grid
        pos: Current position
        move: Direction string
        my_trail: Current trail
        is_boost: Whether this is a boost move (2 steps)
    
    Returns:
        Tuple of (new_grid, new_pos, new_trail)
    """
    # Deep copy grid and trail
    new_grid = [row[:] for row in grid]
    new_trail = list(my_trail)
    new_pos = pos
    
    dx, dy = DIRECTIONS[move]
    num_steps = 2 if is_boost else 1
    
    for _ in range(num_steps):
        x, y = new_pos
        new_pos = torus_wrap((x + dx, y + dy))
        new_trail.append(new_pos)
        new_grid[new_pos[1]][new_pos[0]] = AGENT
    
    return (new_grid, new_pos, new_trail)


def manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate torus-aware Manhattan distance.
    
    Args:
        pos1: First position
        pos2: Second position
    
    Returns:
        Manhattan distance with torus wrapping
    """
    x1, y1 = pos1
    x2, y2 = pos2
    
    dx = min(abs(x1 - x2), WIDTH - abs(x1 - x2))
    dy = min(abs(y1 - y2), HEIGHT - abs(y1 - y2))
    
    return dx + dy


def separated_components(grid: List[List[int]], my_pos: Tuple[int, int], opp_pos: Tuple[int, int], my_trail: List[Tuple[int, int]], opp_trail: List[Tuple[int, int]]) -> bool:
    """
    Check if players are in disconnected regions using BFS.
    
    Args:
        grid: Current board state
        my_pos: My position
        opp_pos: Opponent position
        my_trail: My trail
        opp_trail: Opponent trail
    
    Returns:
        True if players are in separate connected components
    """
    from collections import deque
    
    visited = set()
    queue = deque([my_pos])
    visited.add(torus_wrap(my_pos))
    
    # BFS from my position - only traverse empty cells
    while queue:
        pos = queue.popleft()
        for neighbor in neighbors(pos, grid):
            wrapped = torus_wrap(neighbor)
            if wrapped not in visited:
                x, y = wrapped
                # Check if this cell is empty (not occupied by any trail)
                if grid[y][x] == 0:  # EMPTY
                    visited.add(wrapped)
                    queue.append(wrapped)
    
    # Check if opponent position is reachable
    opp_wrapped = torus_wrap(opp_pos)
    return opp_wrapped not in visited

