"""
Opponent behavior forecasting using online softmax regression.

Learns opponent move probabilities from observed behavior.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from agent_utils import legal_moves, get_current_direction, torus_wrap, HEIGHT, WIDTH
from logic.evaluation import bfs_area
from evaluation import voronoi_score  # Keep old voronoi for now


class SoftmaxOppModel:
    """
    Online softmax regression model for opponent move prediction.
    
    Uses SGD with cross-entropy loss and L2 regularization (λ=1e-4).
    """
    
    def __init__(self, feature_dim: int = 50, learning_rate: float = 0.01, l2_reg: float = 1e-4):
        """
        Initialize opponent model.
        
        Args:
            feature_dim: Dimension of feature vector
            learning_rate: SGD learning rate
            l2_reg: L2 regularization coefficient
        """
        self.feature_dim = feature_dim
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        # Weights: (feature_dim, 4) for [UP, DOWN, LEFT, RIGHT]
        self.weights = np.random.normal(0, 0.01, (feature_dim, 4))
        self.move_history = deque(maxlen=3)  # Last k=3 moves
        self.update_count = 0
        self.pattern_detected = False  # Track if deterministic pattern detected
    
    def features_for_opponent(
        self,
        grid: List[List[int]],
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
        my_trail: List[Tuple[int, int]],
        opp_trail: List[Tuple[int, int]],
        my_boosts: int,
        opp_boosts: int,
        turn_count: int
    ) -> np.ndarray:
        """
        Extract feature vector for opponent move prediction.
        
        Features:
        - dx, dy: Relative position
        - one_hot(last k moves): Move history
        - my_boosts, opp_boosts: Boost counts
        - safe_mask(4): Which directions are safe
        - local_voronoi_diff(4): Voronoi difference for each direction
        - local_floodfill(4): Flood-fill area for each direction
        - cut_proximity(4): Distance to potential cuts
        - turn_parity: Even/odd turn
        
        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = []
        
        # dx, dy (relative position with torus wrapping)
        dx = (opp_pos[0] - my_pos[0]) % WIDTH
        if dx > WIDTH // 2:
            dx -= WIDTH
        dy = (opp_pos[1] - my_pos[1]) % HEIGHT
        if dy > HEIGHT // 2:
            dy -= HEIGHT
        features.extend([dx / WIDTH, dy / HEIGHT])
        
        # One-hot encoding of last k moves (k=3)
        move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        move_history_onehot = [0.0] * (4 * 3)  # 3 moves * 4 directions
        for i, move in enumerate(self.move_history):
            if move in move_map:
                idx = i * 4 + move_map[move]
                move_history_onehot[idx] = 1.0
        features.extend(move_history_onehot)
        
        # Boost counts (normalized)
        features.append(my_boosts / 3.0)
        features.append(opp_boosts / 3.0)
        
        # Safe mask (4 directions)
        current_dir = get_current_direction(opp_trail)
        safe_moves = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
        safe_mask = [
            1.0 if "UP" in safe_moves else 0.0,
            1.0 if "DOWN" in safe_moves else 0.0,
            1.0 if "LEFT" in safe_moves else 0.0,
            1.0 if "RIGHT" in safe_moves else 0.0,
        ]
        features.extend(safe_mask)
        
        # Local Voronoi difference for each direction
        local_voronoi = []
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if direction in safe_moves:
                # Simulate move and compute Voronoi
                from agent_utils import apply_move, DIRECTIONS
                dx, dy = DIRECTIONS[direction]
                test_pos = torus_wrap((opp_pos[0] + dx, opp_pos[1] + dy))
                test_trail = list(opp_trail) + [test_pos]
                try:
                    voronoi = voronoi_score(grid, test_pos, my_pos, test_trail, my_trail)
                    local_voronoi.append(voronoi / 100.0)  # Normalize
                except:
                    local_voronoi.append(0.0)
            else:
                local_voronoi.append(-1.0)  # Invalid move
        features.extend(local_voronoi)
        
        # Local flood-fill for each direction
        local_floodfill = []
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if direction in safe_moves:
                from agent_utils import apply_move, DIRECTIONS
                dx, dy = DIRECTIONS[direction]
                test_pos = torus_wrap((opp_pos[0] + dx, opp_pos[1] + dy))
                test_trail = list(opp_trail) + [test_pos]
                try:
                    area = bfs_area(grid, test_pos, test_trail, my_trail)
                    local_floodfill.append(area / 200.0)  # Normalize
                except:
                    local_floodfill.append(0.0)
            else:
                local_floodfill.append(-1.0)
        features.extend(local_floodfill)
        
        # Cut proximity (distance to potential cutting points)
        cut_proximity = []
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Simplified: distance to center of board
            center_x, center_y = WIDTH // 2, HEIGHT // 2
            from agent_utils import manhattan
            dist = manhattan(opp_pos, (center_x, center_y))
            cut_proximity.append(dist / max(WIDTH, HEIGHT))
        features.extend(cut_proximity)
        
        # Turn parity
        features.append(1.0 if turn_count % 2 == 0 else 0.0)
        
        # Pad or truncate to feature_dim
        feature_vec = np.array(features, dtype=np.float32)
        if len(feature_vec) < self.feature_dim:
            feature_vec = np.pad(feature_vec, (0, self.feature_dim - len(feature_vec)))
        elif len(feature_vec) > self.feature_dim:
            feature_vec = feature_vec[:self.feature_dim]
        
        return feature_vec
    
    def fit_one(self, x: np.ndarray, y: int):
        """
        Online SGD update with cross-entropy loss and L2 regularization.
        
        Adaptive learning rate: increases when deterministic patterns detected.
        
        Args:
            x: Feature vector
            y: True move index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        # Forward pass: compute logits
        logits = x @ self.weights  # (4,)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Detect deterministic pattern: if predicted move has very high probability
        max_prob = np.max(probs)
        if max_prob > 0.85 and np.argmax(probs) == y:
            # Deterministic pattern detected - increase learning rate
            self.pattern_detected = True
            self.learning_rate = self.base_learning_rate * 2.0  # Double learning rate
        elif self.pattern_detected and max_prob < 0.6:
            # Pattern broken - reset learning rate
            self.pattern_detected = False
            self.learning_rate = self.base_learning_rate
        
        # Cross-entropy gradient
        grad = np.zeros_like(self.weights)
        for i in range(4):
            if i == y:
                grad[:, i] = (probs[i] - 1.0) * x
            else:
                grad[:, i] = probs[i] * x
        
        # Add L2 regularization gradient
        grad += self.l2_reg * self.weights
        
        # SGD update
        self.weights -= self.learning_rate * grad
        self.update_count += 1
    
    def predict_proba(
        self,
        grid: List[List[int]],
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
        my_trail: List[Tuple[int, int]],
        opp_trail: List[Tuple[int, int]],
        my_boosts: int,
        opp_boosts: int,
        turn_count: int
    ) -> np.ndarray:
        """
        Predict opponent move probabilities.
        
        Returns:
            Array of shape (4,) with probabilities [pUP, pDOWN, pLEFT, pRIGHT]
            Masked to safe moves and renormalized.
        """
        # Extract features
        x = self.features_for_opponent(
            grid, my_pos, opp_pos, my_trail, opp_trail,
            my_boosts, opp_boosts, turn_count
        )
        
        # Forward pass
        logits = x @ self.weights  # (4,)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Mask to safe moves
        current_dir = get_current_direction(opp_trail)
        safe_moves = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
        move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        masked_probs = np.zeros(4)
        for move in safe_moves:
            if move in move_map:
                idx = move_map[move]
                masked_probs[idx] = probs[idx]
        
        # Renormalize
        total = np.sum(masked_probs)
        if total > 0:
            masked_probs /= total
        else:
            # Fallback: uniform over safe moves
            masked_probs = np.ones(4) / len(safe_moves) if safe_moves else np.array([0.25, 0.25, 0.25, 0.25])
        
        return masked_probs
    
    def update_history(self, move: str):
        """Update move history."""
        self.move_history.append(move)


class EnsembleOppModel:
    """
    Ensemble of opponent models with Bayesian weighting.
    
    Combines SoftmaxOppModel with 4 archetype models:
    - Random: Uniform over safe moves
    - Greedy: Maximizes own area
    - Voronoi: Maximizes territory
    - Minimax: Worst-case for opponent
    """
    
    def __init__(self):
        self.softmax_model = SoftmaxOppModel()
        self.weights = np.array([0.5, 0.125, 0.125, 0.125, 0.125])  # Softmax + 4 archetypes
        self.move_history = deque(maxlen=3)
    
    def predict_proba(
        self,
        grid: List[List[int]],
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
        my_trail: List[Tuple[int, int]],
        opp_trail: List[Tuple[int, int]],
        my_boosts: int,
        opp_boosts: int,
        turn_count: int
    ) -> np.ndarray:
        """Weighted average of model predictions."""
        # Get softmax prediction
        softmax_probs = self.softmax_model.predict_proba(
            grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count
        )
        
        # Get archetype predictions (simplified)
        current_dir = get_current_direction(opp_trail)
        safe_moves = legal_moves(grid, opp_pos, current_dir, opp_trail, my_trail)
        move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        # Random archetype
        random_probs = np.ones(4) / len(safe_moves) if safe_moves else np.array([0.25, 0.25, 0.25, 0.25])
        
        # Greedy archetype (simplified: prefer moves with more area)
        greedy_probs = self._greedy_probs(grid, opp_pos, opp_trail, my_trail, safe_moves, move_map)
        
        # Voronoi archetype
        voronoi_probs = self._voronoi_probs(grid, opp_pos, my_pos, opp_trail, my_trail, safe_moves, move_map)
        
        # Minimax archetype (worst for us)
        minimax_probs = self._minimax_probs(grid, opp_pos, my_pos, opp_trail, my_trail, safe_moves, move_map)
        
        # Weighted average
        ensemble_probs = (
            self.weights[0] * softmax_probs +
            self.weights[1] * random_probs +
            self.weights[2] * greedy_probs +
            self.weights[3] * voronoi_probs +
            self.weights[4] * minimax_probs
        )
        
        # Renormalize
        ensemble_probs /= np.sum(ensemble_probs)
        return ensemble_probs
    
    def _greedy_probs(self, grid, opp_pos, opp_trail, my_trail, safe_moves, move_map):
        """Greedy archetype: maximize own area."""
        probs = np.zeros(4)
        scores = []
        
        for move in safe_moves:
            from agent_utils import apply_move
            new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
            area = bfs_area(new_grid, new_pos, new_trail, my_trail)
            scores.append((area, move))
        
        if scores:
            max_score = max(s[0] for s in scores)
            for area, move in scores:
                if move in move_map:
                    idx = move_map[move]
                    probs[idx] = area / max_score if max_score > 0 else 1.0 / len(safe_moves)
        
        if np.sum(probs) == 0:
            probs = np.ones(4) / len(safe_moves) if safe_moves else np.array([0.25, 0.25, 0.25, 0.25])
        else:
            probs /= np.sum(probs)
        
        return probs
    
    def _voronoi_probs(self, grid, opp_pos, my_pos, opp_trail, my_trail, safe_moves, move_map):
        """Voronoi archetype: maximize territory."""
        probs = np.zeros(4)
        scores = []
        
        for move in safe_moves:
            from agent_utils import apply_move
            new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
            voronoi = voronoi_score(new_grid, new_pos, my_pos, new_trail, my_trail)
            scores.append((voronoi, move))
        
        if scores:
            max_score = max(s[0] for s in scores)
            min_score = min(s[0] for s in scores)
            for voronoi, move in scores:
                if move in move_map:
                    idx = move_map[move]
                    normalized = (voronoi - min_score) / (max_score - min_score + 1e-6)
                    probs[idx] = normalized
        
        if np.sum(probs) == 0:
            probs = np.ones(4) / len(safe_moves) if safe_moves else np.array([0.25, 0.25, 0.25, 0.25])
        else:
            probs /= np.sum(probs)
        
        return probs
    
    def _minimax_probs(self, grid, opp_pos, my_pos, opp_trail, my_trail, safe_moves, move_map):
        """Minimax archetype: worst-case for us."""
        probs = np.zeros(4)
        scores = []
        
        for move in safe_moves:
            from agent_utils import apply_move
            new_grid, new_pos, new_trail = apply_move(grid, opp_pos, move, opp_trail, False)
            # Evaluate from our perspective (opponent wants to minimize our score)
            from evaluation import evaluate_position
            score = evaluate_position(new_grid, my_pos, new_pos, my_trail, new_trail)
            scores.append((score, move))
        
        if scores:
            min_score = min(s[0] for s in scores)  # Worst for us
            max_score = max(s[0] for s in scores)
            for score, move in scores:
                if move in move_map:
                    idx = move_map[move]
                    # Higher probability for moves that minimize our score
                    normalized = 1.0 - (score - min_score) / (max_score - min_score + 1e-6)
                    probs[idx] = normalized
        
        if np.sum(probs) == 0:
            probs = np.ones(4) / len(safe_moves) if safe_moves else np.array([0.25, 0.25, 0.25, 0.25])
        else:
            probs /= np.sum(probs)
        
        return probs
    
    def fit_one(self, x: np.ndarray, y: int):
        """Update softmax model."""
        self.softmax_model.fit_one(x, y)
    
    def update_history(self, move: str):
        """Update move history."""
        self.move_history.append(move)
        self.softmax_model.update_history(move)

