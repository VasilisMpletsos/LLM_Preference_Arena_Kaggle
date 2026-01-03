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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

sys.path.append(str(Path.cwd().parent))
import bitsandbytes as bnb
import loralib as lora
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import CLASSIFICATION_PROMPT, CosineLearningDecay, remove_extra_brackets

if __name__ == "__main__":
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 64
    writer = SummaryWriter("./logs/qwen_llm_finetune")

    # Load multiple CSV files
    df = datasets.load_dataset(
        "csv", data_files={"train": "../data/train.csv", "test": "../data/test.csv"}
    )

    # Qwen/Qwen3-4B-Instruct-2507
    # Qwen/Qwen3-0.6B

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model max length is {tokenizer.model_max_length} characters.")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        device_map="cuda",
        num_labels=3,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        modules_to_save=["score"],
    )

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    lora.mark_only_lora_as_trainable(model)
    for param in model.score.parameters():
        param.requires_grad = True

    # Enable gradient checkpointing on raw pytorch
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    # model.config.use_cache = False

    model.print_trainable_parameters()
    ## Fix Dataset

    def fix_dataset(row):
        cleaned_prompt = remove_extra_brackets(row["prompt"])
        cleaned_response_a = remove_extra_brackets(row["response_a"])
        cleaned_response_b = remove_extra_brackets(row["response_b"])

        full_prompt = CLASSIFICATION_PROMPT.format(
            prompt=cleaned_prompt,
            response_a=cleaned_response_a,
            response_b=cleaned_response_b,
        )

        tokenized = tokenizer(full_prompt)

        winner = [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]]

        return {**tokenized, "winner": winner, "length": len(tokenized["input_ids"])}

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

    df = df.filter(
        lambda batch: np.array(batch["length"]) <= 12000, batched=True
    ).remove_columns(["length"])

    train_val_split = df["train"].train_test_split(test_size=0.05, seed=42)
    df["train"] = train_val_split["train"]
    df["validation"] = train_val_split["test"]
    df = df.with_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    train_dataloader = DataLoader(
        df["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        df["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator
    )
    # # Experiment

    # data = next(iter(train_dataloader))
    # data["input_ids"].shape
    # data["input_ids"] = data["input_ids"].to("cuda")
    # with torch.no_grad():
    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         outputs = model(data["input_ids"]).logits
    #         sums = data["attention_mask"].sum(dim=1) - 1
    # preds = outputs[torch.arange(2), sums, :]
    # torch.argmax(preds, dim=1)
    # torch.argmax(data['winner'], dim=1)
    # # Training

    # model = torch.compile(model)
    STARTING_LEARNING_RATE = 1e-4
    # optimizer = AdamW(model.parameters(), lr=STARTING_LEARNING_RATE, weight_decay=0.01)
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=STARTING_LEARNING_RATE, weight_decay=0.01
    )
    # optimizer = bnb.optim.Adam8bit(model.parameters(), min_8bit_size=16384)
    EPOCHS = 10
    scheduler = CosineLearningDecay(
        max_lr=STARTING_LEARNING_RATE,
        min_lr=1e-6,
        optimizer=optimizer,
        max_steps=250_000,
        warmup_steps=1_000,
    )
    loss_fn = CrossEntropyLoss()

    torch.set_float32_matmul_precision("medium")

    grad_steps_corrects = 0
    grad_steps_count = 0

    train_size = len(train_dataloader)
    best_validation_loss = float("inf")
    best_model_weights = None

    for epoch in range(EPOCHS):
        model.train()
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
            scheduler.update_lr((epoch * train_size) + (step + 1))
            data = {key: value.to("cuda") for key, value in data.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = outputs = model(data["input_ids"]).logits
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    _, true_labels = torch.max(data["winner"], 1)
                    examples_count = data["input_ids"].size(0)
                    correct_count = (predicted == true_labels).sum().item()
                    grad_steps_count += examples_count
                    total_count += examples_count
                    total_correct += correct_count
                    grad_steps_corrects += correct_count
                    if (step + 1) % (GRADIENT_ACCUMULATION_STEPS // 4) == 0:
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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar(
                    "Accuracy/train", accuracy, (epoch * train_size) + (step + 1)
                )

            # torch.cuda.empty_cache()

            if (step + 1) % (train_size // 5) == 0 and step != 0:
                model.eval()
                correct = 0
                total = 0
                val_loss = 0
                for data in validation_bar:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            data = {
                                key: value.to("cuda") for key, value in data.items()
                            }
                            outputs = model(data["input_ids"]).logits
                            _, predicted = torch.max(outputs, 1)
                            _, true_labels = torch.max(data["winner"], 1)
                            val_loss += loss_fn(outputs, true_labels).item()
                            total += true_labels.size(0)
                            correct += (predicted == true_labels).sum().item()

                accuracy = 100 * (correct / total)
                print(f"Validation Accuracy: {accuracy:.2f}%")
                print(f"Validation Loss: {val_loss:.2f}%")
                writer.add_scalar(
                    "Accuracy/validation", accuracy, (epoch * train_size) + (step + 1)
                )
                model.train()

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    best_model_weights = model.state_dict()
                    print(
                        f"New best model found with val accuracy {accuracy:.2f}% & loss {val_loss:.4f}."
                    )
                    torch.save(
                        model.state_dict(),
                        "./models/best_qwen_llm_finetune_model.pth",
                    )

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Training Accuracy: {100 * (total_correct / total_count):.2f}%"
        )
