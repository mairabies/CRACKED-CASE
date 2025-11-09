# Case Closed - ImGoatedAndIKIt Agent

## Quick Start

**For submission, see the `submission/` folder which contains all required files.**

The main agent code is in `submission/agent.py` - this is what will be evaluated.

## Project Structure

```
CRACKED-CASE/
├── submission/          # SUBMISSION FOLDER - All required files here
│   ├── agent.py         # Main agent (REQUIRED)
│   ├── requirements.txt # Dependencies (REQUIRED)
│   ├── Dockerfile       # Container config (REQUIRED)
│   ├── case_closed_game.py
│   ├── agent_utils.py
│   ├── boosts.py
│   ├── logic/           # Evaluation, search, regime detection
│   ├── models/          # Opponent modeling
│   └── README.md        # Detailed submission documentation
│
├── other/               # Development/testing files (not submitted)
│   ├── testing/        # Test scripts
│   ├── training/        # RL training code
│   ├── analysis/        # Tournament analysis
│   ├── documentation/   # Development docs
│   └── strategies/      # Alternative strategies
│
└── README.md           # This file
```

## Submission

**Submit the `submission/` folder to Devpost/GitHub.**

All required files are in `submission/`:
- `agent.py` - ImGoatedAndIKIt agent
- `requirements.txt` - All dependencies
- `Dockerfile` - Container configuration
- `case_closed_game.py` - Game logic
- All supporting modules

## Key Restrictions

- **CPU-only PyTorch**: Uses `torch` (CPU version) - no CUDA
- **5GB Docker Limit**: Minimal dependencies
- **No Tensorflow/JAX**: Only PyTorch for opponent modeling
- **Decision Time**: Optimized for <50ms per move

## Agent: ImGoatedAndIKIt

**ImGoatedAndIKIt** is a heuristic-based agent featuring:

- **Opponent Modeling**: Online softmax regression learns patterns in real-time
- **Adaptive Strategy Selection**: Detects game regimes and switches strategies
- **Counter-Strategies**: Automatically detects and counters FloodFill/Voronoi
- **Expectimax Search**: Multi-turn lookahead with CVaR risk management
- **Defensive Safety**: Prioritizes moves that maximize future options

### Performance

- **Win Rate vs FloodFill**: 100% (tested on 500 games)
- **Win Rate vs Voronoi**: 100% (tested on 500 games)
- **Win Rate vs Beam**: 100% (tested on 500 games)
- **Self-Play**: Stable, no crashes

## Detailed Documentation

See `submission/README.md` for:
- Complete agent architecture
- Strategy details
- API endpoints
- Testing results
- Docker build instructions

## Local Testing

```bash
# Install dependencies
pip install -r submission/requirements.txt

# Run agent
cd submission
python agent.py
```

## Docker Build

```bash
cd submission
docker build -t case-closed-agent .
docker run -p 5008:8080 case-closed-agent
```

## Contact

**Participant**: Maira(bies) Athar  
**Agent Name**: ImGoatedAndIKIt

---

**Note**: Always test your container before submitting. The Dockerfile has been tested and verified.
