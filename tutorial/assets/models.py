from datetime import datetime

import evaluate
import numpy as np
import torch
from dagster import (
    AutoMaterializePolicy,
    DataVersion,
    FreshnessPolicy,
    MetadataValue,
    Output,
    asset,
    observable_source_asset,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

from tutorial.resources.resources import BaseLLM, TrainingResource, AcceleratorResource

@asset(
    description="HuggingFace Tokenizer",
)
def tokenizer(BaseLLM: BaseLLM):
    my_tokenizer = BaseLLM.PretrainedTokenizer()
    return Output(
        my_tokenizer,
        metadata={"model_name": MetadataValue.text(BaseLLM.model_name)},
    )


@asset(
    description="Base HuggingFace large language model for fine-tuning",
)
def base_llm(
    BaseLLM: BaseLLM, accelerator: AcceleratorResource
) -> Output[torch.nn.Module]:

    model = BaseLLM.LoraModel()
    # Place model on accelerator
    model = model.to(accelerator.get_device())

    return Output(
        model,
        metadata={
            "model_name": MetadataValue.text(BaseLLM.model_name),
            "trainable_parameters": MetadataValue.text(
                model.print_trainable_parameters()
            ),
        },
    )

def evaluate_peft_model(sample, model, tokenizer, max_target_length=512): 
    # generate summary
    outputs = model.generate(
            input_ids=sample["input_ids"].unsqueeze(0).cpu(),
            do_sample=True,
            top_p=0.9,
            max_new_tokens=512,)
    prediction = tokenizer.decode(
            outputs[0].detach().cpu().numpy(), skip_special_tokens=True
        )
    # decode eval sample
    labels = np.where(
            sample["labels"] != -100, sample["labels"], tokenizer.pad_token_id
        )
    # Some simple post-processing
    labels = tokenizer.decode(labels, skip_special_tokens=True)
    return prediction, labels




@asset(
    description="A LoRA fine-tuned HuggingFace large language model",
)
def finetuned_llm(
    TrainingResource: TrainingResource,
    base_llm: torch.nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    accelerator: AcceleratorResource,
) -> Output[torch.nn.Module]:
    # Place model on accelerator
    base_llm = base_llm.to(accelerator.get_device())
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=base_llm,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    training_args = TrainingResource.training_args()

    trainer = Seq2SeqTrainer(
        model=base_llm,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    _finetuned_llm = trainer.model
    _finetuned_llm.eval()


    print("Peft model loaded")

    metric = evaluate.load("rouge")
    predictions, references = [], []

    model = _finetuned_llm.to(accelerator.get_device())
    for sample in val_dataset.with_format("torch"):

        prediction, labels = evaluate_peft_model(sample, model, tokenizer, max_target_length=512)
        predictions.append(prediction)
        references.append(labels)
    rogue = metric.compute(
    predictions=predictions, references=references, use_stemmer=True)

    eval_rouge_scores = {
        "rogue1": rogue["rouge1"] * 100,
        "rouge2": rogue["rouge2"] * 100,
        "rougeL": rogue["rougeL"] * 100,
        "rougeLsum": rogue["rougeLsum"] * 100,
    }

    eval_metric_all = {**eval_metrics, **eval_rouge_scores}

    return Output(
        _finetuned_llm, 
        metadata={
            name: MetadataValue.float(value) for name, value in eval_metric_all.items()
        },
    )


