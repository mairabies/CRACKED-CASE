"""
Expectimax search with CVaR (Conditional Value at Risk) for risk-aware decisions.

Uses stochastic opponent model instead of minimax worst-case assumption.
"""

import time
import random
import numpy as np
from typing import List, Tuple, Optional, Callable
from agent_utils import (
    legal_moves, apply_move, get_current_direction, torus_wrap,
    HEIGHT, WIDTH
)
from logic.evaluation import score_state
from search import simulate_game_step  # Import from original search.py


def legal_moves_fast(grid: List[List[int]], pos: Tuple[int, int], current_dir: Optional[str], my_trail: List[Tuple[int, int]], opp_trail: List[Tuple[int, int]]) -> List[str]:
    """Fast legal moves check (re-export from agent_utils)."""
    from agent_utils import legal_moves
    return legal_moves(grid, pos, current_dir, my_trail, opp_trail)


def one_ply_expectimax(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    opp_probs: np.ndarray,
    score_fn: Callable,
    n_samples: int = 8,
    cvar_alpha: float = 0.2
) -> str:
    """
    One-ply expectimax: sample opponent replies and average with CVaR.
    
    Args:
        grid: Current board grid
        my_pos: My position
        opp_pos: Opponent position
        my_trail: My trail
        opp_trail: Opponent trail
        opp_probs: Opponent move probabilities [pUP, pDOWN, pLEFT, pRIGHT]
        score_fn: Scoring function
        n_samples: Number of opponent move samples
        cvar_alpha: CVaR quantile (0.2 = worst 20%)
    
    Returns:
        Best move direction string
    """
    current_dir = get_current_direction(my_trail)
    my_moves = legal_moves_fast(grid, my_pos, current_dir, my_trail, opp_trail)
    
    if not my_moves:
        return "UP"
    
    if len(my_moves) == 1:
        return my_moves[0]
    
    move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    inv_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    best_move = my_moves[0]
    best_score = float('-inf')
    
    for my_move in my_moves:
        scores = []
        
        # Sample opponent moves according to probabilities
        for _ in range(n_samples):
            # Sample opponent move
            opp_idx = np.random.choice(4, p=opp_probs)
            opp_move = inv_map[opp_idx]
            
            # Check if move is legal
            opp_current_dir = get_current_direction(opp_trail)
            opp_legal = legal_moves_fast(grid, opp_pos, opp_current_dir, opp_trail, my_trail)
            
            if opp_move not in opp_legal:
                # Resample or skip
                if opp_legal:
                    opp_move = random.choice(opp_legal)
                else:
                    continue
            
            # Simulate
            try:
                new_g, new_m_pos, new_o_pos, new_m_trail, new_o_trail, m_alive, o_alive = simulate_game_step(
                    grid, my_pos, opp_pos, my_move, opp_move, my_trail, opp_trail
                )
                
                if not m_alive:
                    score = -1000.0
                elif not o_alive:
                    score = 1000.0
                else:
                    score = score_fn(new_g, new_m_pos, new_o_pos, new_m_trail, new_o_trail)
                
                scores.append(score)
            except:
                continue
        
        if not scores:
            continue
        
        # Compute expected value and CVaR
        scores_array = np.array(scores)
        ev = np.mean(scores_array)
        
        # CVaR: average of worst (1-alpha) fraction
        n_worst = max(1, int(len(scores) * cvar_alpha))
        worst_scores = np.partition(scores_array, n_worst - 1)[:n_worst]
        cvar = np.mean(worst_scores)
        
        # Combined score: EV - tau * CVaR (risk-averse)
        tau = 0.3  # Risk aversion parameter
        combined_score = ev - tau * (ev - cvar)  # Penalize downside
        
        if combined_score > best_score:
            best_score = combined_score
            best_move = my_move
    
    return best_move


def beam_search_expectimax(
    grid: List[List[int]],
    my_pos: Tuple[int, int],
    opp_pos: Tuple[int, int],
    my_trail: List[Tuple[int, int]],
    opp_trail: List[Tuple[int, int]],
    opp_probs: np.ndarray,
    score_fn: Callable,
    depth: int = 2,
    beam: int = 3,
    time_ms: int = 40,
    n_samples: int = 8,
    tau: float = 0.3
) -> str:
    """
    Short beam search with expectimax and CVaR.
    
    Args:
        grid: Current board grid
        my_pos: My position
        opp_pos: Opponent position
        my_trail: My trail
        opp_trail: Opponent trail
        opp_probs: Opponent move probabilities
        score_fn: Scoring function
        depth: Search depth (2-3)
        beam: Beam width
        time_ms: Hard time limit
        n_samples: Opponent move samples per node
        tau: Risk aversion (higher = more risk-averse)
    
    Returns:
        Best move direction string
    """
    start_time = time.time()
    time_limit = time_ms / 1000.0
    
    current_dir = get_current_direction(my_trail)
    my_moves = legal_moves_fast(grid, my_pos, current_dir, my_trail, opp_trail)
    
    if not my_moves:
        return "UP"
    
    if len(my_moves) == 1:
        return my_moves[0]
    
    move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    inv_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    best_move = my_moves[0]
    best_score = float('-inf')
    
    # Iterative deepening
    for current_depth in range(1, depth + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize beam: (my_move, my_pos, opp_pos, my_trail, opp_trail, grid, score)
        beam_nodes = [(move, my_pos, opp_pos, my_trail, opp_trail, grid, 0.0) for move in my_moves]
        
        for d in range(current_depth):
            if time.time() - start_time > time_limit:
                break
            
            next_beam = []
            
            for my_move, m_pos, o_pos, m_trail, o_trail, g, _ in beam_nodes:
                # Sample opponent moves
                opp_current_dir = get_current_direction(o_trail)
                opp_legal = legal_moves_fast(g, o_pos, opp_current_dir, o_trail, m_trail)
                
                if not opp_legal:
                    # Opponent has no moves
                    score = score_fn(g, m_pos, o_pos, m_trail, o_trail)
                    next_beam.append((my_move, m_pos, o_pos, m_trail, o_trail, g, score))
                    continue
                
                # Sample opponent moves
                opp_scores = []
                for _ in range(n_samples):
                    opp_idx = np.random.choice(4, p=opp_probs)
                    opp_move = inv_map[opp_idx]
                    
                    if opp_move not in opp_legal:
                        if opp_legal:
                            opp_move = random.choice(opp_legal)
                        else:
                            continue
                    
                    try:
                        new_g, new_m_pos, new_o_pos, new_m_trail, new_o_trail, m_alive, o_alive = simulate_game_step(
                            g, m_pos, o_pos, my_move, opp_move, m_trail, o_trail
                        )
                        
                        if not m_alive:
                            score = -1000.0
                        elif not o_alive:
                            score = 1000.0
                        else:
                            score = score_fn(new_g, new_m_pos, new_o_pos, new_m_trail, new_o_trail)
                        
                        opp_scores.append(score)
                    except:
                        continue
                
                if not opp_scores:
                    score = score_fn(g, m_pos, o_pos, m_trail, o_trail)
                    next_beam.append((my_move, m_pos, o_pos, m_trail, o_trail, g, score))
                    continue
                
                # Compute EV and CVaR
                scores_array = np.array(opp_scores)
                ev = np.mean(scores_array)
                
                # CVaR: worst-case average
                n_worst = max(1, int(len(opp_scores) * 0.2))
                worst_scores = np.partition(scores_array, n_worst - 1)[:n_worst]
                cvar = np.mean(worst_scores)
                
                # Risk-adjusted score: EV - tau * max_loss
                max_loss = ev - np.min(scores_array)
                combined_score = ev - tau * max_loss
                
                # Use average state (simplified)
                next_beam.append((my_move, m_pos, o_pos, m_trail, o_trail, g, combined_score))
            
            # Prune to top beam by EV - tau * max_loss
            next_beam.sort(key=lambda x: x[6], reverse=True)
            beam_nodes = next_beam[:beam]
        
        # Evaluate final positions
        if beam_nodes:
            beam_nodes.sort(key=lambda x: x[6], reverse=True)
            best_move = beam_nodes[0][0]
            best_score = beam_nodes[0][6]
    
    return best_move

