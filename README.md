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

## Running censorship web API

start the web service by running

```bash
python bin/web.py
```

and test it with `curl`

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"I hate you!"}' http://localhost:8001/censorship_status
curl -X POST -H "Content-Type: application/json" -d '{"text":"I love you!"}' http://localhost:8001/censorship_status
```
