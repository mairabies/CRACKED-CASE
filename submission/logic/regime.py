"""
Regime detection and hybrid move selection.

Determines game state (separated vs competitive) and selects appropriate strategy.
"""

from typing import List, Tuple, Optional
from collections import deque
import numpy as np
from agent_utils import (
    legal_moves, get_current_direction, torus_wrap, HEIGHT, WIDTH, EMPTY, manhattan, apply_move
)
from logic.evaluation import bfs_area, score_state, voronoi_diff
from logic.search import one_ply_expectimax, beam_search_expectimax


def separated_components(grid: List[List[int]], my_pos: Tuple[int, int], opp_pos: Tuple[int, int]) -> bool:
    """
    Check if players are in disconnected regions using two BFS.
    
    Args:
        grid: Board grid
        my_pos: My position
        opp_pos: Opponent position
    
    Returns:
        True if players are in separate connected components
    """
    # BFS from my position
    my_visited = set()
    my_queue = deque([my_pos])
    my_visited.add(torus_wrap(my_pos))
    
    while my_queue:
        pos = my_queue.popleft()
        from agent_utils import neighbors
        for neighbor in neighbors(pos, grid):
            n_wrapped = torus_wrap(neighbor)
            if n_wrapped not in my_visited:
                x, y = n_wrapped
                if grid[y][x] == EMPTY:
                    my_visited.add(n_wrapped)
                    my_queue.append(neighbor)
    
    # BFS from opponent position
    opp_visited = set()
    opp_queue = deque([opp_pos])
    opp_visited.add(torus_wrap(opp_pos))
    
    while opp_queue:
        pos = opp_queue.popleft()
        from agent_utils import neighbors
        for neighbor in neighbors(pos, grid):
            n_wrapped = torus_wrap(neighbor)
            if n_wrapped not in opp_visited:
                x, y = n_wrapped
                if grid[y][x] == EMPTY:
                    opp_visited.add(n_wrapped)
                    opp_queue.append(neighbor)
    
    # Check if either can reach the other's current cell
    my_wrapped = torus_wrap(my_pos)
    opp_wrapped = torus_wrap(opp_pos)
    
    # Separated if neither can reach the other's position
    return (opp_wrapped not in my_visited) and (my_wrapped not in opp_visited)


def detect_floodfill_opponent(
    grid: List[List[int]],
    opp_pos: Tuple[int, int],
    opp_trail: List[Tuple[int, int]],
    my_trail: List[Tuple[int, int]],
    opp_probs: np.ndarray
) -> bool:
    """
    Detect if opponent is using FloodFill strategy (always maximizes area).
    
    Returns True if opponent behavior matches FloodFill pattern.
    """
    # FloodFill is deterministic: always picks move that maximizes BFS area
    # Check if opponent's move probabilities are highly concentrated on area-maximizing moves
    
    current_dir = get_current_direction(opp_trail)
    opp_legal = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
    
    if len(opp_legal) < 2:
        return False
    
    # Calculate area for each legal move
    area_scores = []
    for move in opp_legal:
        new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
        area = bfs_area(new_grid, new_pos)
        area_scores.append((area, move))
    
    # Find move with max area (what FloodFill would choose)
    max_area_move = max(area_scores, key=lambda x: x[0])[1]
    move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    
    # Check if opponent probabilities are highly concentrated on max-area move
    if max_area_move in move_map:
        max_area_idx = move_map[max_area_move]
        max_area_prob = opp_probs[max_area_idx]
        
        # FloodFill would have >80% probability on the max-area move
        return max_area_prob > 0.8
    
    return False


def detect_voronoi_opponent(
    grid: List[List[int]],
    opp_pos: Tuple[int, int],
    my_pos: Tuple[int, int],
    opp_trail: List[Tuple[int, int]],
    my_trail: List[Tuple[int, int]],
    opp_probs: np.ndarray
) -> bool:
    """
    Detect if opponent is using Voronoi strategy (maximizes territory).
    
    Returns True if opponent behavior matches Voronoi pattern.
    """
    current_dir = get_current_direction(opp_trail)
    opp_legal = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
    
    if len(opp_legal) < 2:
        return False
    
    # Calculate Voronoi score for each legal move
    voronoi_scores = []
    for move in opp_legal:
        new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
        voronoi = voronoi_diff(new_grid, new_pos, my_pos)
        voronoi_scores.append((voronoi, move))
    
    # Find move with max Voronoi (what Voronoi agent would choose)
    max_voronoi_move = max(voronoi_scores, key=lambda x: x[0])[1]
    move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    
    if max_voronoi_move in move_map:
        max_voronoi_idx = move_map[max_voronoi_move]
        max_voronoi_prob = opp_probs[max_voronoi_idx]
        
        # Voronoi agent would have >70% probability on max-Voronoi move
        return max_voronoi_prob > 0.7
    
    return False


def counter_floodfill_move(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    moves: List[str],
    score_fn
) -> str:
    """
    Counter-strategy against FloodFill: predict their move and cut them off.
    """
    # Predict FloodFill's next move (maximize their area)
    current_dir = get_current_direction(opp_trail)
    opp_legal = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
    
    predicted_opp_move = None
    max_opp_area = -1
    
    for move in opp_legal:
        new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
        area = bfs_area(new_grid, new_pos)
        if area > max_opp_area:
            max_opp_area = area
            predicted_opp_move = move
    
    if predicted_opp_move is None:
        # Fallback to normal strategy
        return one_ply_expectimax(
            grid, my_pos, opp_pos, my_trail, opp_trail,
            np.array([0.25, 0.25, 0.25, 0.25]), score_fn, n_samples=8, cvar_alpha=0.2
        )
    
    # Predict opponent's next position
    from agent_utils import DIRECTIONS, torus_wrap
    dx, dy = DIRECTIONS[predicted_opp_move]
    predicted_opp_pos = torus_wrap((opp_pos[0] + dx, opp_pos[1] + dy))
    
    # Find our move that:
    # 1. Gets closer to predicted position (to cut them off)
    # 2. Reduces their future area
    # 3. Maintains our own area
    best_move = moves[0]
    best_score = float('-inf')
    
    for move in moves:
        new_grid, new_pos, new_trail = apply_move(grid, my_pos, move, my_trail, False)
        
        # Distance to predicted opponent position (closer is better for cutting)
        dist_to_predicted = manhattan(new_pos, predicted_opp_pos)
        
        # Our area after move
        my_area = bfs_area(new_grid, new_pos)
        
        # Opponent's area after their predicted move (simulate)
        opp_new_grid, opp_new_pos, opp_new_trail = apply_move(
            new_grid, predicted_opp_pos, predicted_opp_move, 
            list(opp_trail) + [predicted_opp_pos], False
        )
        opp_area = bfs_area(opp_new_grid, opp_new_pos)
        
        # Score: maximize our area, minimize their area, get closer to cut
        score = my_area - 1.5 * opp_area - 0.5 * dist_to_predicted
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move


def counter_voronoi_move(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    moves: List[str],
    score_fn
) -> str:
    """
    Counter-strategy against Voronoi: be more aggressive, reduce their territory.
    """
    # More aggressive scoring: prioritize reducing opponent territory
    def aggressive_score_fn(g, m_pos, o_pos, m_trail, o_trail):
        my_area = bfs_area(g, m_pos)
        opp_area = bfs_area(g, o_pos)
        voronoi = voronoi_diff(g, m_pos, o_pos)
        
        # Aggressive: heavily penalize opponent territory
        return 1.2 * my_area - 1.5 * opp_area + 0.8 * voronoi
    
    # Use deeper search with aggressive scoring
    return beam_search_expectimax(
        grid, my_pos, opp_pos, my_trail, opp_trail,
        np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform (Voronoi is deterministic)
        aggressive_score_fn, depth=3, beam=4, time_ms=40
    )


def choose_move_hybrid(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    opp_probs: np.ndarray,
    score_fn,
    time_ms: int = 40,
    use_expectimax: bool = True
) -> str:
    """
    Hybrid move selection with regime detection.
    
    Strategy:
    - If only one safe move → take it
    - If separated → maximize own area (FloodFill)
    - Else → expectimax/beam with forecaster + CVaR
    
    Args:
        grid: Board grid
        my_pos: My position
        opp_pos: Opponent position
        my_trail: My trail
        opp_trail: Opponent trail
        opp_probs: Opponent move probabilities
        score_fn: Scoring function
        time_ms: Time limit
        use_expectimax: Use expectimax (True) or beam search (False)
    
    Returns:
        Best move direction string
    """
    current_dir = get_current_direction(my_trail)
    moves = legal_moves(grid, my_pos, current_dir, my_trail, opp_trail)
    
    if not moves:
        return "UP"  # Emergency fallback
    
    # If only one safe move, take it
    if len(moves) == 1:
        return moves[0]
    
    # Defensive check: if opponent is very close (< 3 cells), prioritize safety
    opp_distance = manhattan(my_pos, opp_pos)
    if opp_distance <= 3:
        # In danger zone - prioritize moves that maximize future options
        best_move = moves[0]
        best_future_moves = 0
        
        for move in moves:
            new_grid, new_pos, new_trail = apply_move(grid, my_pos, move, my_trail, False)
            new_dir = get_current_direction(new_trail)
            future_moves = len(legal_moves(new_grid, new_pos, new_dir, new_trail, opp_trail))
            
            if future_moves > best_future_moves:
                best_future_moves = future_moves
                best_move = move
        
        return best_move
    
    # Check if separated
    separated = separated_components(grid, my_pos, opp_pos)
    
    if separated:
        # Maximize own area (FloodFill strategy)
        best_move = moves[0]
        best_area = 0
        
        for move in moves:
            new_grid, new_pos, new_trail = apply_move(grid, my_pos, move, my_trail, False)
            area = bfs_area(new_grid, new_pos)
            
            if area > best_area:
                best_area = area
                best_move = move
        
        return best_move
    else:
        # Competitive situation: detect opponent strategy and adapt
        # Detect FloodFill: opponent consistently maximizes area
        is_floodfill = detect_floodfill_opponent(grid, opp_pos, opp_trail, my_trail, opp_probs)
        
        if is_floodfill:
            # Counter FloodFill: predict their move and cut them off
            return counter_floodfill_move(
                grid, my_pos, opp_pos, my_trail, opp_trail, moves, score_fn
            )
        
        # Detect Voronoi: opponent consistently maximizes territory
        is_voronoi = detect_voronoi_opponent(grid, opp_pos, my_pos, opp_trail, my_trail, opp_probs)
        
        if is_voronoi:
            # Counter Voronoi: be more aggressive, reduce their territory
            return counter_voronoi_move(
                grid, my_pos, opp_pos, my_trail, opp_trail, moves, score_fn
            )
        
        # Default: use expectimax/beam with forecaster
        if use_expectimax:
            return one_ply_expectimax(
                grid, my_pos, opp_pos, my_trail, opp_trail,
                opp_probs, score_fn, n_samples=12, cvar_alpha=0.15  # More samples, less risk-averse
            )
        else:
            return beam_search_expectimax(
                grid, my_pos, opp_pos, my_trail, opp_trail,
                opp_probs, score_fn, depth=3, beam=4, time_ms=time_ms  # Deeper search
            )


