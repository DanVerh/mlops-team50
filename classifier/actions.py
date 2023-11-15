import os
import sys

import torch
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from classifier.dataset import Dataset
from classifier.model import Model

model_hidden_size_mapping = {"bert-base-uncased": 768, "lyeonii/bert-small": 512}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_model(model_version="lyeonii/bert-small") -> Model:
    hidden_size = model_hidden_size_mapping[model_version]
    return Model(
        model_version=model_version, hidden_size=hidden_size, device=select_device()
    )


def load_trained_model(model_version="lyeonii/bert-small") -> Model:
    model = get_model(model_version)
    model.model.load_state_dict(
        torch.load(
            os.path.join(model.model_path, model.model_file),
            map_location=torch.device(select_device()),
        )
    )
    return model


def train(model_version="lyeonii/bert-small", epochs: int = 30, batch_size: int = 64):
    model = get_model(model_version)
    tokenizer = BertTokenizerFast.from_pretrained(model_version)

    train_dataloader = Dataset().prepare_dataloaders(
        "train", tokenizer, model.hidden_size, batch_size=batch_size
    )
    val_dataloader = Dataset().prepare_dataloaders(
        "val", tokenizer, model.hidden_size, batch_size=batch_size
    )
    model.train(train_dataloader, val_dataloader, epochs)


def evaluate(model_version="lyeonii/bert-small"):
    model = load_trained_model(model_version)
    tokenizer = BertTokenizerFast.from_pretrained(model_version)

    dataset = Dataset()
    test_seq, test_mask, test_y = dataset.prepare_tensors_from_df(
        dataset.preprocess()["test"].iloc[:100, :],
        tokenizer=tokenizer,
        hidden_size=model.hidden_size,
    )

    test_preds = model.predict(test_seq, test_mask)
    print(model.prepare_classification_report(test_y, test_preds))


def predict(input_text: list[str], model: Model):
    tokenizer = BertTokenizerFast.from_pretrained(model.model_version)

    dataset = Dataset()
    test_seq, test_mask = dataset.prepare_tensors(
        input_text, tokenizer=tokenizer, hidden_size=model.hidden_size
    )

    return model.predict(test_seq, test_mask)


def print_predictions(input_text: list[str], model_version="lyeonii/bert-small"):
    model = load_trained_model(model_version)
    results = predict(input_text, model)
    for t, res in zip(input_text, results):
        print(
            f"---------------------------------------------------------------------------\nInput text: {t}\nIs unacceptable: {res}"
        )


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


def predict_random_n(n, max_len):
    print_predictions(get_n_examples(n, max_len))
