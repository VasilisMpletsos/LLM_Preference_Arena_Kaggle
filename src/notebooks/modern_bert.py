# %%
import datasets
import seaborn as sns

sns.set_style("whitegrid")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

sys.path.append(str(Path.cwd().parent))
from transformers import DataCollatorWithPadding

from utils import CosineLearningDecay, remove_extra_brackets

if __name__ == "__main__":
    # %%
    writer = SummaryWriter("./logs/modern_bert")

    # %%
    # Load multiple CSV files
    df = datasets.load_dataset(
        "csv", data_files={"train": "../data/train.csv", "test": "../data/test.csv"}
    )

    # %%
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    print(f"Model max length is {tokenizer.model_max_length} characters.")
    model_classification = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=3
    )
    model_classification = model_classification.to("cuda", torch.bfloat16)
    model_maskedLM = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")

    # %%
    def fix_dataset(row):
        cleaned_prompt = remove_extra_brackets(row["prompt"])
        cleaned_response_a = remove_extra_brackets(row["response_a"])
        cleaned_response_b = remove_extra_brackets(row["response_b"])

        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id

        p_ids = tokenizer(cleaned_prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(cleaned_response_a, add_special_tokens=False)["input_ids"]
        b_ids = tokenizer(cleaned_response_b, add_special_tokens=False)["input_ids"]

        # Structure: [CLS] + Prompt + [SEP] + A + [SEP] + B + [SEP]
        input_ids = [cls_id] + p_ids + [sep_id] + a_ids + [sep_id] + b_ids + [sep_id]

        winner = [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]]

        return {
            "input_ids": input_ids,
            "winner": winner,
            "length": len(input_ids),  # easy filtering later
        }

    # %%
    df = df.map(fix_dataset, batched=False).remove_columns(
        [
            "id",
            "model_a",
            "model_b",
            "prompt",
            "response_a",
            "response_b",
            "winner_model_a",
            "winner_model_b",
            "winner_tie",
        ]
    )

    # %%
    df = df.filter(
        lambda batch: np.array(batch["length"]) <= 8192, batched=True
    ).remove_columns(["length"])

    # %%
    train_val_split = df["train"].train_test_split(test_size=0.05, seed=42)
    df["train"] = train_val_split["train"]
    df["validation"] = train_val_split["test"]
    df = df.with_format("torch")
    # Initialize the collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    train_dataloader = DataLoader(
        df["train"], batch_size=4, shuffle=True, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        df["validation"], batch_size=4, shuffle=False, collate_fn=data_collator
    )

    # %%
    # next(iter(train_dataloader))

    # %%
    model_classification = torch.compile(model_classification)

    # %%
    STARTING_LEARNING_RATE = 1e-4
    optimizer = AdamW(
        model_classification.parameters(), lr=STARTING_LEARNING_RATE, weight_decay=0.01
    )
    EPOCHS = 100
    scheduler = CosineLearningDecay(
        max_lr=STARTING_LEARNING_RATE,
        min_lr=1e-6,
        optimizer=optimizer,
        max_steps=650000,
        warmup_steps=1000,
    )
    loss_fn = CrossEntropyLoss()
    GRADIENT_ACCUMULATION_STEPS = 32

    torch.set_float32_matmul_precision("medium")

    grad_steps_corrects = 0
    grad_steps_count = 0

    train_size = len(train_dataloader)

    for epoch in range(EPOCHS):
        model_classification.train()
        total_loss = 0
        total_correct = 0
        total_count = 0
        optimizer.zero_grad()

        train_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True, position=0
        )
        validation_bar = tqdm(
            val_dataloader,
            desc=f"Validation {epoch + 1}/{EPOCHS}",
            leave=False,
            position=0,
        )
        for step, data in enumerate(train_bar):
            data = {key: value.to("cuda") for key, value in data.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model_classification(data["input_ids"]).logits
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    _, true_labels = torch.max(data["winner"], 1)
                    examples_count = data["input_ids"].size(0)
                    correct_count = (predicted == true_labels).sum().item()
                    grad_steps_count += examples_count
                    total_count += examples_count
                    total_correct += correct_count
                    grad_steps_corrects += correct_count
                    if (step + 1) % 10 == 0:
                        train_bar.set_postfix(
                            {
                                "Prediction": f"{predicted.cpu().tolist()} | {true_labels.cpu().tolist()}"
                            }
                        )

                loss = loss_fn(outputs, true_labels)

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                accuracy = 100 * (grad_steps_corrects / grad_steps_count)
                grad_steps_corrects = 0
                grad_steps_count = 0

                train_bar.set_postfix({"accuracy": f"{(accuracy):.2f}%"})
                # torch.nn.utils.clip_grad_norm_(model_classification.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.update_lr((epoch * train_size) + (step + 1))
                optimizer.zero_grad()

                writer.add_scalar(
                    "Accuracy/train", accuracy, (epoch * train_size) + (step + 1)
                )

            if step % 4000 == 0 and step != 0:
                model_classification.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_data in validation_bar:
                        val_data = {
                            key: value.to("cuda") for key, value in val_data.items()
                        }
                        outputs = model_classification(val_data["input_ids"]).logits
                        _, predicted = torch.max(outputs, 1)
                        _, true_labels = torch.max(val_data["winner"], 1)
                        total += true_labels.size(0)
                        correct += (predicted == true_labels).sum().item()

                accuracy = 100 * (correct / total)
                print(f"Validation Accuracy: {accuracy:.2f}%")
                writer.add_scalar(
                    "Accuracy/validation", accuracy, (epoch * train_size) + (step + 1)
                )
                model_classification.train()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Training Accuracy: {100 * (total_correct / total_count):.2f}%"
        )
