import os
import sys

import typer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import classifier.actions as actions

app = typer.Typer()


@app.command()
def train(model_version="lyeonii/bert-small", epochs: int = 30, batch_size: int = 64):
    actions.train(model_version, epochs, batch_size)


@app.command()
def evaluate(model_version="lyeonii/bert-small"):
    actions.evaluate(model_version)


@app.command()
def predict_random_n(n, max_len):
    actions.predict_random_n(n, max_len)


if __name__ == "__main__":
    app()
