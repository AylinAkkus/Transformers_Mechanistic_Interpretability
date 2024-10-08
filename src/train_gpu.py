#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, DataCollatorWithPadding, Trainer, EarlyStoppingCallback
import pandas as pd
from datasets import Dataset
from torch_datasets import TopKDataset, DisplaceDataset
import torch
import numpy as np

PROJECT_NAME = "1l1h_displace_3_range_64"
N_LAYERS = 1
N_HEADS = 1
LOG_DIR = f"logs/{PROJECT_NAME}"
USE_DATASET_GENERATION = True
# DATA_TRAIN_PATH = "data/top_two_max_len_3_range_64_train.csv"
# DATA_EVAL_PATH = "data/top_two_max_len_3_range_64_val.csv"
TRAIN_EPOCHS = 4
MODEL_NAME = "distilbert-base-uncased"
RESUME_FROM_CHECKPOINT = False
EARLY_STOPPING_PATIENCE = 5
# LR = 2e-5
LR = 1e-5
LOGGING_PER_STEPS = 50
SAVE_PER_STEPS = 250
SAVE_TOTAL_LIMIT = 2
PER_DEVICE_BATCH_SIZE = 32

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def tokenize_and_prepare_labels(dataset):
    # Tokenize the text and labels
    tokenized_dataset = tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=128)
    tokenized_labels = tokenizer(dataset['labels'], add_special_tokens=False)['input_ids']

    # tokenized_dataset contains 'input_ids' (token of input texts padded to max_length) and 'attention_mask'
    # Initialize labels with -100
    labels = [[-100] * len(tokenized_input) for tokenized_input in tokenized_dataset['input_ids']] 

    # Replace -100 with the label ID at the masked position
    for i, input_ids in enumerate(tokenized_dataset['input_ids']):
        # Find the index of the [MASK] token
        mask_indices = [j for j, x in enumerate(input_ids) if x == tokenizer.mask_token_id]
        # Replace -100 with the token ID for the correct label
        for k, mask_index in enumerate(mask_indices):
            labels[i][mask_index] = tokenized_labels[i][k]

    tokenized_dataset['labels'] = labels
    return tokenized_dataset

def tokenize_and_prepare_labels_for_torch_dataset(dataset):
    # Tokenize the text and labels
    tokenized_dataset = tokenizer(dataset.text, padding="max_length", truncation=True, max_length=128)
    tokenized_labels = tokenizer(dataset.labels, add_special_tokens=False)['input_ids']

    # tokenized_dataset contains 'input_ids' (token of input texts padded to max_length) and 'attention_mask'
    # Initialize labels with -100
    labels = [[-100] * len(tokenized_input) for tokenized_input in tokenized_dataset['input_ids']] 

    # Replace -100 with the label ID at the masked position
    for i, input_ids in enumerate(tokenized_dataset['input_ids']):
        # Find the index of the [MASK] token
        mask_indices = [j for j, x in enumerate(input_ids) if x == tokenizer.mask_token_id]
        # Replace -100 with the token ID for the correct label
        for k, mask_index in enumerate(mask_indices):
            labels[i][mask_index] = tokenized_labels[i][k]

    tokenized_dataset['labels'] = labels
    return tokenized_dataset

if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)

    config = AutoConfig.from_pretrained(MODEL_NAME, n_heads=N_HEADS, n_layers=N_LAYERS)
    model = AutoModelForMaskedLM.from_config(config)
    model.to(device)  # Move the model to the GPU if available
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=LOG_DIR,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        evaluation_strategy='steps',
        logging_strategy='steps',
        logging_steps=LOGGING_PER_STEPS,
        report_to='none',  # Disable logging to wandb
        save_steps=SAVE_PER_STEPS,  # Save checkpoint every 500 steps
        save_total_limit=SAVE_TOTAL_LIMIT,  # Only keep the last 2 checkpoints
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,  # Resume from the last checkpoint
        seed=42,
        load_best_model_at_end=True,
        fp16=True  # Enable mixed precision training for better performance on GPUs
    )

    if USE_DATASET_GENERATION:
        data_train = DisplaceDataset(num_samples=50000, length=5).data  # Adjust these lines to the corresponding dataset
        data_eval = DisplaceDataset(num_samples=10000, length=5).data
    else:
        data_train = pd.read_csv(DATA_TRAIN_PATH, dtype=str)
        data_eval = pd.read_csv(DATA_EVAL_PATH, dtype=str)
    dataset_train = Dataset.from_pandas(data_train)
    dataset_eval = Dataset.from_pandas(data_eval)
    tokenized_dataset_train = dataset_train.map(tokenize_and_prepare_labels, batched=True)
    tokenized_dataset_eval = dataset_eval.map(tokenize_and_prepare_labels, batched=True)
    tokenized_dataset_train = tokenized_dataset_train.shuffle(seed=42)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    eval_result = trainer.evaluate()
    print(eval_result)
    trainer.save_model(LOG_DIR + "/trained")
