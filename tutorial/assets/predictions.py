import torch
from dagster import AutoMaterializePolicy, MetadataValue, Output, asset

from tutorial.resources.resources import AcceleratorResource 


@asset(
    description="Predictions on Slack user questions",
)
def predictions(
    finetuned_llm: torch.nn.Module,
    tokenizer,
    inference_dataset,
    accelerator: AcceleratorResource,
):

    finetuned_llm = finetuned_llm.to(accelerator.get_device())
    predictions = []
    questions = []
    with torch.inference_mode():
        
        for sample in inference_dataset.with_format('torch', device=accelerator.get_device()):
            max_target_length = 512
            outputs = finetuned_llm.generate(
                input_ids=sample["input_ids"].unsqueeze(0).cpu(),
                do_sample=True,
                top_p=0.9,
                max_new_tokens=max_target_length,
            )
            prediction = tokenizer.decode(
                outputs[0].detach().cpu().numpy(), skip_special_tokens=True
            )
            predictions.append(prediction)
            questions.append(
                tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            )

    return Output(
        predictions,
        metadata={
            "predictions": MetadataValue.json(predictions),
            "questions": MetadataValue.json(questions),
        },
    )


