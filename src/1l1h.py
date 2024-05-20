#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, DataCollatorWithPadding, Trainer
import pandas as pd
from datasets import Dataset
import wandb

LOG_DIR = "logs/simple_model"
DATA_TRAIN_PATH = "data/train_simple.csv"
DATA_EVAL_PATH = "data/val_simple.csv"
TRAIN_EPOCHS = 20000
MODEL_NAME = "distilbert-base-uncased"
RESUME_FROM_CHECKPOINT = True


def tokenize_and_prepare_labels(dataset):
    # Tokenize the text
    tokenized_dataset = tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=128)

    # Initialize labels with -100
    labels = [[-100] * len(tokenized_input) for tokenized_input in tokenized_dataset['input_ids']]

    # Replace -100 with the label ID at the masked position
    for i, (input_ids, label) in enumerate(zip(tokenized_dataset['input_ids'], labels)):
        # Find the index of the [MASK] token
        mask_index = input_ids.index(tokenizer.mask_token_id)
        # Replace -100 with the token ID for the correct label
        labels[i][mask_index] = tokenizer.convert_tokens_to_ids(dataset['labels'][i])

    tokenized_dataset['labels'] = labels
    return tokenized_dataset


if __name__ == "__main__":
    wandb.init(project="1l1h_simple")
    config = AutoConfig.from_pretrained(MODEL_NAME, n_heads=1, n_layers=1)
    # print(config)
    model = AutoModelForMaskedLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=LOG_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=TRAIN_EPOCHS,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        logging_steps=10,
        report_to='wandb',  # Enable logging to wandb
        run_name='my_training_run',  # Name of the run in wandb
        save_steps=1000,  # Save checkpoint every 500 steps
        save_total_limit=2,  # Only keep the last 2 checkpoints
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,  # Resume from the last checkpoint
        seed=42,
    )

    data_train = pd.read_csv(DATA_TRAIN_PATH)
    dataset_train = Dataset.from_pandas(data_train)
    data_eval = pd.read_csv(DATA_EVAL_PATH)
    dataset_eval = Dataset.from_pandas(data_eval)
    tokenized_dataset_train = dataset_train.map(tokenize_and_prepare_labels, batched=True)
    tokenized_dataset_eval = dataset_eval.map(tokenize_and_prepare_labels, batched=True)
    tokenized_dataset_train = tokenized_dataset_train.shuffle(seed=42)
    print(tokenized_dataset_train)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    trainer.save_model(LOG_DIR + "/trained")

    wandb.finish()