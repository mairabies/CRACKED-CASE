"""
Speed boost decision logic.

Determines when it's safe and advantageous to use a speed boost.
"""

from typing import List, Tuple
from agent_utils import apply_move, is_safe, torus_wrap, manhattan, DIRECTIONS


def should_boost(
    grid: List[List[int]],
    move: str,
    my_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_pos: Tuple[int, int],
    opp_trail: List[Tuple[int, int]],
    boosts_left: int,
    opp_distance: int
) -> bool:
    """
    Determine if using a boost is safe and advantageous.
    
    Args:
        grid: Current board grid
        move: Proposed move direction
        my_pos: My current position
        my_trail: My trail positions
        opp_trail: Opponent trail positions
        boosts_left: Number of boosts remaining
        opp_distance: Distance to opponent
    
    Returns:
        True if boost should be used
    """
    if boosts_left <= 0:
        return False
    
    # Check if both steps of the boost are safe
    dx, dy = DIRECTIONS[move]
    step1_pos = torus_wrap((my_pos[0] + dx, my_pos[1] + dy))
    step2_pos = torus_wrap((step1_pos[0] + dx, step1_pos[1] + dy))
    
    # Check first step
    if not is_safe(step1_pos, grid, my_trail, opp_trail):
        return False
    
    # Check second step (simulate first step to get updated trail)
    temp_grid, temp_pos, temp_trail = apply_move(grid, my_pos, move, my_trail, False)
    if not is_safe(step2_pos, temp_grid, temp_trail, opp_trail):
        return False
    
    # Heuristic: use boost if:
    # 1. Opponent is close (escape situation) - DEFENSIVE PRIORITY
    # 2. We're in a good position to cut off opponent
    # 3. We have multiple boosts left (don't waste the last one early)
    
    # Defensive: Escape from close opponent
    if opp_distance <= 2 and boosts_left >= 1:
        return True
    
    # Moderate danger: escape if we have boosts
    if opp_distance <= 3 and boosts_left >= 2:
        return True
    
    # Use boost if we're in a favorable position and have boosts to spare
    if opp_distance <= 5 and boosts_left >= 3:
        return True
    
    return False


