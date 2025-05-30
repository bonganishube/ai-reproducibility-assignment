
# Reproducibility Assignment: Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics

This repository contains our reproduction of the paper *"Utilising uncertainty for efficient learning of likely-admissible heuristics"* by Ofir Maron and Benjamin Rosman (ICAPS 2020).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Team](#team)
- [Report](#report)
- [License](#license)

## Introduction

We reproduce the experiments from the paper that proposes:
1. A method for learning likely-admissible heuristics using uncertainty estimates
2. Techniques to improve heuristic search efficiency
3. Evaluation across three planning environments

Our implementation includes:
- Core algorithm reimplementation in Python
- Experiments on all original environments
- Additional hyperparameter analysis

## Installation

### Prerequisites
- Python ≥ 3.8
- pip

### Setup
```bash
git clone https://github.com/[your-username]/heuristic-learning-reproduction.git
cd heuristic-learning-reproduction

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
