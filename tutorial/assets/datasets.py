import json
import datasets
import numpy as np
import pandas as pd
from dagster import AutoMaterializePolicy, Config, MetadataValue, Output, asset
from sklearn.model_selection import train_test_split
from tutorial.resources.resources import RawDatasetConfig 

@asset(description="Slack Q&A user data")
def raw_datasets(config: RawDatasetConfig):

    with open(config.path_to_slack_messages, "r") as f:
        data = json.load(f)

    format_data = []
    for sample in data:
        format_data.append(
            {
                "input_text": sample[config.input_text_key],
                "output_text": sample[config.output_text_key],
            }
        )
    format_data = pd.DataFrame(format_data)
    
    train, test= train_test_split(format_data, 
                                   random_state=config.seed, 
                                   train_size=config.train_size)
    ## split the test set into a validation set and inference set 
    validation, inference = train_test_split(test, 
                                random_state=config.seed, 
                                train_size=.8)
    
    dataset_train = datasets.Dataset.from_pandas(train)
    dataset_validation = datasets.Dataset.from_pandas(validation)
    dataset_inference = datasets.Dataset.from_pandas(inference)

    dataset = datasets.DatasetDict({"train": dataset_train, "validation": dataset_validation, "inference": dataset_inference})
    
    return Output(dataset, metadata= {"Train dataset size": len(dataset_train), "Test dataset size": len(dataset_validation), "Inference dataset size": len(dataset_inference)})


@asset(
    description="Processed and deanonymized Q&A data",
)
def processed_datasets(
    tokenizer, raw_datasets):
    tokenized_inputs = datasets.concatenate_datasets([raw_datasets["train"], raw_datasets["validation"],  raw_datasets["inference"]]).map(
        lambda x: tokenizer(x["input_text"], truncation=True),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]
    max_source_lengths = int(np.percentile(input_lengths, 95))
    print(f"Max source lengths: {max_source_lengths}")

    tokenized_targets = datasets.concatenate_datasets([raw_datasets["train"], raw_datasets["validation"],  raw_datasets["inference"]]).map(
        lambda x: tokenizer(x["output_text"], truncation=True),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_lengths = int(np.percentile(target_lengths, 95))
    print(f"Max target lengths: {max_target_lengths}")

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = [item for item in sample["input_text"]]
        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_lengths, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=sample["output_text"],
            max_length=max_target_lengths,
            padding=padding,
            truncation=True,
        )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = datasets.DatasetDict()
    tokenized_dataset["train"] = raw_datasets["train"].map(
        preprocess_function, batched=True, remove_columns=["input_text", "output_text"]
    )
    tokenized_dataset["validation"] = raw_datasets["validation"].map(
        preprocess_function, batched=True, remove_columns=["input_text", "output_text"]
    )
    tokenized_dataset["inference"] = raw_datasets["inference"].map(
        preprocess_function, batched=True, remove_columns=["input_text", "output_text"]
    )
    return tokenized_dataset

@asset(
    description="Training dataset",
)
def train_dataset(processed_datasets: datasets.DatasetDict):
    dataset = processed_datasets["train"]
    return Output(
        dataset,
        metadata={
            "size (bytes)": MetadataValue.int(dataset.size_in_bytes),
            "info": MetadataValue.text(dataset.info.description),
            "len": MetadataValue.int(len(dataset)),
        },
    )


@asset(
    description="Validation dataset",
)
def val_dataset(processed_datasets: datasets.DatasetDict):
    dataset = processed_datasets["validation"]
    return Output(
        dataset,
        metadata={
            "size (bytes)": MetadataValue.int(dataset.size_in_bytes),
            "info": MetadataValue.text(dataset.info.description),
            "len": MetadataValue.int(len(dataset)),
        },
    )

@asset(
    description="Inference dataset",
)
def inference_dataset(processed_datasets: datasets.DatasetDict):
    dataset = processed_datasets["inference"]
    return Output(
        dataset,
        metadata={
            "size (bytes)": MetadataValue.int(dataset.size_in_bytes),
            "info": MetadataValue.text(dataset.info.description),
            "len": MetadataValue.int(len(dataset)),
        },
    )
