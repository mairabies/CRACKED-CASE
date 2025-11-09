import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult
from agent_utils import parse_state, manhattan
from models.opponent_forecaster import SoftmaxOppModel
from logic.regime import choose_move_hybrid
from logic.evaluation import score_state
from boosts import should_boost

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "Maira(bies) Athar"
AGENT_NAME = "ImGoatedAndIKIt"

# Global opponent model (persists across ticks)
opponent_model = SoftmaxOppModel()
last_opp_move = None


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    global last_opp_move
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    # Track opponent's move by comparing trail positions
    if opp_trail := data.get("agent1_trail" if LAST_POSTED_STATE.get("player_number") == 2 else "agent2_trail"):
        prev_trail = LAST_POSTED_STATE.get("agent1_trail" if LAST_POSTED_STATE.get("player_number") == 2 else "agent2_trail", [])
        if len(prev_trail) >= 1 and len(opp_trail) >= 2:
            # Infer move from trail change
            from agent_utils import get_current_direction
            last_opp_move = get_current_direction(opp_trail)
            if last_opp_move:
                opponent_model.update_history(last_opp_move)
    
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        state["player_number"] = player_number
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = my_agent.boosts_remaining
        turn_count = state.get("turn_count", 0)
   
    # Parse game state
    grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count, _ = parse_state(state)
    
    # Update opponent model if we observed their last move
    global last_opp_move, opponent_model
    if last_opp_move and opp_trail:
        # Extract features and update model
        try:
            x = opponent_model.features_for_opponent(
                grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count
            )
            move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
            if last_opp_move in move_map:
                opponent_model.fit_one(x, move_map[last_opp_move])
        except:
            pass
    
    # Predict opponent move probabilities
    opp_probs = opponent_model.predict_proba(
        grid, my_pos, opp_pos, my_trail, opp_trail, my_boosts, opp_boosts, turn_count
    )
    
    # Score function with defensive weights (prioritize safety and area control)
    # Adjust weights based on game phase
    opp_distance = manhattan(my_pos, opp_pos)
    if opp_distance <= 3:
        # Close combat: prioritize safety and mobility
        def score_fn(g, m_pos, o_pos, m_trail, o_trail):
            return score_state(g, m_pos, o_pos, m_trail, o_trail, 
                             alpha=1.2, beta=0.6, gamma=0.8, delta=0.6, kappa=0.5)
    else:
        # Normal play: balanced weights
        def score_fn(g, m_pos, o_pos, m_trail, o_trail):
            return score_state(g, m_pos, o_pos, m_trail, o_trail, 
                             alpha=1.0, beta=0.8, gamma=0.6, delta=0.4)
    
    # Get move using hybrid strategy
    move = choose_move_hybrid(
        grid, my_pos, opp_pos, my_trail, opp_trail,
        opp_probs, score_fn, time_ms=40, use_expectimax=True
    )
    
    # Check if boost should be used
    if boosts_remaining > 0:
        opp_distance = manhattan(my_pos, opp_pos)
        if should_boost(
            grid, move, my_pos, my_trail, opp_pos, opp_trail, my_boosts, opp_distance
        ):
            move = f"{move}:BOOST"

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    global last_opp_move
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        last_opp_move = None  # Reset for next game
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
