"""
Optimized evaluation heuristics with fast BFS and Voronoi calculations.

Uses NumPy arrays for efficient visited tracking and multi-source BFS.
"""

import numpy as np
from typing import List, Tuple
from collections import deque
from agent_utils import HEIGHT, WIDTH, EMPTY, AGENT, torus_wrap, neighbors, is_safe


def bfs_area(grid: List[List[int]], pos: Tuple[int, int], max_radius: int = 14) -> int:
    """
    Fast BFS area calculation using NumPy array for visited tracking.
    
    Args:
        grid: Board grid
        pos: Starting position
        max_radius: Maximum BFS radius (default 14 for performance)
    
    Returns:
        Number of reachable empty cells
    """
    visited = np.zeros((HEIGHT, WIDTH), dtype=bool)
    queue = deque([(pos, 0)])
    wrapped = torus_wrap(pos)
    visited[wrapped[1], wrapped[0]] = True
    count = 0
    
    while queue:
        pos, dist = queue.popleft()
        wrapped = torus_wrap(pos)
        x, y = wrapped
        
        # Limit radius for performance
        if dist > max_radius:
            continue
        
        # Count if empty
        if grid[y][x] == EMPTY:
            count += 1
        
        # Explore neighbors
        for neighbor in neighbors(pos, grid):
            n_wrapped = torus_wrap(neighbor)
            nx, ny = n_wrapped
            
            if not visited[ny, nx] and grid[ny][nx] == EMPTY:
                visited[ny, nx] = True
                queue.append((neighbor, dist + 1))
    
    return count


def voronoi_diff(grid: List[List[int]], my_pos: Tuple[int, int], opp_pos: Tuple[int, int]) -> float:
    """
    Fast Voronoi difference using 2 multi-source BFS.
    
    Args:
        grid: Board grid
        my_pos: My position
        opp_pos: Opponent position
    
    Returns:
        Territory difference (my_territory - opp_territory)
    """
    my_dist = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    opp_dist = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    
    my_wrapped = torus_wrap(my_pos)
    opp_wrapped = torus_wrap(opp_pos)
    
    my_dist[my_wrapped[1], my_wrapped[0]] = 0
    opp_dist[opp_wrapped[1], opp_wrapped[0]] = 0
    
    my_queue = deque([(my_pos, 0)])
    opp_queue = deque([(opp_pos, 0)])
    
    # Expand both BFS trees
    while my_queue or opp_queue:
        # My BFS
        if my_queue:
            pos, dist = my_queue.popleft()
            wrapped = torus_wrap(pos)
            x, y = wrapped
            
            if grid[y][x] == EMPTY:
                for neighbor in neighbors(pos, grid):
                    n_wrapped = torus_wrap(neighbor)
                    nx, ny = n_wrapped
                    
                    if grid[ny][nx] == EMPTY and my_dist[ny, nx] == np.inf:
                        my_dist[ny, nx] = dist + 1
                        my_queue.append((neighbor, dist + 1))
        
        # Opponent BFS
        if opp_queue:
            pos, dist = opp_queue.popleft()
            wrapped = torus_wrap(pos)
            x, y = wrapped
            
            if grid[y][x] == EMPTY:
                for neighbor in neighbors(pos, grid):
                    n_wrapped = torus_wrap(neighbor)
                    nx, ny = n_wrapped
                    
                    if grid[ny][nx] == EMPTY and opp_dist[ny, nx] == np.inf:
                        opp_dist[ny, nx] = dist + 1
                        opp_queue.append((neighbor, dist + 1))
    
    # Count territory
    my_territory = np.sum((my_dist < opp_dist) & (my_dist < np.inf))
    opp_territory = np.sum((opp_dist < my_dist) & (opp_dist < np.inf))
    
    return float(my_territory - opp_territory)


def chokepoint_penalty(grid: List[List[int]], next_pos: Tuple[int, int], threshold: int = 10) -> float:
    """
    Two-step local flood-fill to detect chokepoints.
    
    Args:
        grid: Board grid
        next_pos: Position after move
        threshold: Minimum reachable cells to avoid penalty
    
    Returns:
        Penalty score (higher = worse chokepoint)
    """
    # Quick two-step flood-fill
    reachable = bfs_area(grid, next_pos, max_radius=2)
    
    if reachable < threshold:
        return 10.0 - reachable  # Higher penalty for tighter chokepoints
    return 0.0


def crash_imminent(grid: List[List[int]], next_pos: Tuple[int, int], my_trail: List[Tuple[int, int]], opp_trail: List[Tuple[int, int]]) -> bool:
    """
    Check if moving to next_pos will cause immediate crash.
    
    Args:
        grid: Board grid
        next_pos: Position to check
        my_trail: My trail
        opp_trail: Opponent trail
    
    Returns:
        True if crash is imminent
    """
    wrapped = torus_wrap(next_pos)
    x, y = wrapped
    
    # Check grid
    if grid[y][x] == AGENT:
        return True
    
    # Check trails
    if wrapped in my_trail or wrapped in opp_trail:
        return True
    
    return False


def score_state(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.6,
    delta: float = 0.4,
    kappa: float = 0.3
) -> float:
    """
    Comprehensive state evaluation with tunable constants.
    
    Score = α * my_area - β * opp_area + γ * voronoi_diff - δ * chokepoint - κ * crash_risk
    
    Args:
        grid: Board grid
        my_pos: My position
        opp_pos: Opponent position
        my_trail: My trail
        opp_trail: Opponent trail
        alpha: Weight for my area
        beta: Weight for opponent area
        gamma: Weight for Voronoi difference
        delta: Weight for chokepoint penalty
        kappa: Weight for crash risk
    
    Returns:
        Evaluation score (higher = better)
    """
    my_area = bfs_area(grid, my_pos)
    opp_area = bfs_area(grid, opp_pos)
    voronoi = voronoi_diff(grid, my_pos, opp_pos)
    chokepoint = chokepoint_penalty(grid, my_pos)
    crash_risk = 1.0 if crash_imminent(grid, my_pos, my_trail, opp_trail) else 0.0
    
    score = (
        alpha * my_area -
        beta * opp_area +
        gamma * voronoi -
        delta * chokepoint -
        kappa * crash_risk * 100.0  # Large penalty for crashes
    )
    
    return score


