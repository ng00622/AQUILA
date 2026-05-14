# AQUILA: AI-driven Quantum-safe Integrated LEO Architecture

Simulation code for IEEE ICMLCN 2026 paper.

## Overview

AQUILA implements PPO-based orchestration for hybrid QKD-PQC key management in LEO satellite constellations.

## Quick Start

```bash
pip install numpy matplotlib
python simulation/aquila_simulation.py
```

## Results

| Strategy | QKD (Mb) | PQC (Mb) | ITS % |
|----------|----------|----------|-------|
| QKD-Only | 16.93 | 0.00 | 100% |
| PQC-Only | 0.00 | 2412.92 | 0% |
| Static-Hybrid | 16.93 | 2326.23 | 0.7% |
| **PPO (AQUILA)** | 16.93 | 926.18 | **60.3%** |

## Authors

- Neha (6GIC, University of Surrey)
- Dr. Mohammad Shojafar (Supervisor)

## License

MIT License
