import os
import sys

import torch
import typer
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from classifier.dataset import Dataset
from classifier.model import Model

app = typer.Typer()

model_hidden_size_mapping = {"bert-base-uncased": 768, "lyeonii/bert-small": 512}

def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@app.command()
def train(model_version="lyeonii/bert-small", epochs: int = 30, batch_size: int = 64):
    hidden_size = model_hidden_size_mapping[model_version]
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    device = select_device()

    train_dataloader = Dataset().prepare_dataloaders(
        "train", tokenizer, hidden_size, batch_size=batch_size
    )
    val_dataloader = Dataset().prepare_dataloaders(
        "val", tokenizer, hidden_size, batch_size=batch_size
    )

    Model(model_version=model_version, hidden_size=hidden_size, device=device).train(
        train_dataloader, val_dataloader, epochs
    )


@app.command()
def evaluate(model_version="lyeonii/bert-small"):
    hidden_size = model_hidden_size_mapping[model_version]
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    device = select_device()

    dataset = Dataset()
    test_seq, test_mask, test_y = dataset.prepare_tensors_from_df(
        dataset.preprocess()["test"].iloc[:100, :],
        tokenizer=tokenizer,
        hidden_size=hidden_size,
    )

    model = Model(
        model_version=model_version, hidden_size=hidden_size, device=device
    )
    model.model.load_state_dict(
        torch.load(
            os.path.join(model.model_path, model.model_file),
            map_location=torch.device(device),
        )
    )

    test_preds = model.predict(test_seq, test_mask)
    print(model.prepare_classification_report(test_y, test_preds))


@app.command()
def predict(input_text: list[str], model_version="lyeonii/bert-small"):
    hidden_size = model_hidden_size_mapping[model_version]
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    device = select_device()

    dataset = Dataset()
    test_seq, test_mask = dataset.prepare_tensors(
        input_text, tokenizer=tokenizer, hidden_size=hidden_size
    )

    model = Model(
        model_version=model_version, hidden_size=hidden_size, device=device
    )
    model.model.load_state_dict(
        torch.load(
            os.path.join(model.model_path, model.model_file),
            map_location=torch.device(device),
        )
    )

    results = model.predict(test_seq, test_mask)
    for t, res in zip(input_text, results):
        print(
            f"---------------------------------------------------------------------------\nInput text: {t}\nIs unacceptable: {res}"
        )


@app.command()
def get_n_examples(n, max_len):
    n = int(n)
    dataset = Dataset()
    examples = (
        dataset.preprocess()["test"]
        .query(f"text_len < {max_len}")
        .sample(n)["text"]
        .to_list()
    )
    return examples


@app.command()
def predict_random_n(n, max_len):
    predict(get_n_examples(n, max_len))


if __name__ == "__main__":
    app()
