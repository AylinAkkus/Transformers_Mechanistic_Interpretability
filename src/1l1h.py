from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM, pipeline, DistilBertConfig, TrainingArguments, DataCollatorWithPadding, Trainer
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


'''
Make a dataset file into huggingface dataset format of fill-mask task
and split it into train and test
'''
def make_dataset(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None, names=["first", "second", "mask"])
    data['input_text'] = data['first'].astype(str) + " " + data['second'].astype(str) + " [MASK]"
    data['labels'] = data['mask']
    dataset = Dataset.from_pandas(data[['input_text', 'labels']])
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

# Tokenize the data
def tokenize_dataset(dataset):
    return tokenizer(dataset['input_text'], padding="max_length", truncation=True)
    # labels = dataset['labels']
    # return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}

def tokenize_and_prepare_labels(examples):
    # Tokenize the texts
    tokenized_inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=128)

    # Create labels array initialized with -100 (ignore index)
    labels = [-100] * len(tokenized_inputs['input_ids'][0])

    # Assign the correct label for the masked token
    mask_index = tokenized_inputs['input_ids'][0].index(tokenizer.mask_token_id)
    labels[mask_index] = examples['labels'][0]

    # Ensure labels have the same structure as inputs
    tokenized_inputs['labels'] = [labels]
    return tokenized_inputs

# Applying tokenization and label preparation
# tokenized_datasets = dataset.map(tokenize_and_prepare_labels, batched=True)



if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    config = AutoConfig.from_pretrained(model_name, n_heads=1, n_layers=1)
    # print(config)
    model = AutoModelForMaskedLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    training_args = TrainingArguments(
        output_dir="huggingface/logs",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
    )




    # dataset = make_dataset("huggingface/data/train.txt")

    # tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    # tokenized_dataset = dataset.map(tokenize_and_prepare_labels, batched=True)

    data = pd.read_csv("huggingface/data/train.csv")
    dataset = Dataset.from_pandas(data)
    def tokenize_and_prepare_labels(examples):
        # Tokenize the text
        tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        # Initialize labels with -100
        labels = [[-100] * len(tokenized_input) for tokenized_input in tokenized_inputs['input_ids']]

        # Replace -100 with the label ID at the masked position
        for i, (input_ids, label) in enumerate(zip(tokenized_inputs['input_ids'], labels)):
            # Find the index of the [MASK] token
            mask_index = input_ids.index(tokenizer.mask_token_id)
            # Replace -100 with the token ID for the correct label
            labels[i][mask_index] = tokenizer.convert_tokens_to_ids(examples['labels'][i])

        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    tokenized_dataset = dataset.map(tokenize_and_prepare_labels, batched=True)

    print(tokenized_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=tokenized_dataset["train"],
        # eval_dataset=tokenized_dataset["test"],
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
