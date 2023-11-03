# Training project for MLOps course

## Model

Bert pretrained

## Dataset

Commonsense morality dataset
'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar'

## Fine-tuning task

Text classification

## Prerequisites

1. python 3.10, you can install one on Mac with `brew install python@3.10`
2. poetry, can be installed with `pipx install poetry`
3. black formatter `pipx install black`, optionally you can also install a black extenstion for you favorite IDE

## Dev Environment Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
poetry install --no-root
```

## Model Training

```bash
python bin/console.py train
```
