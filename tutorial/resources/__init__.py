from tutorial.resources.resources import BaseLLM, TrainingResource, AcceleratorResource


accelerator = AcceleratorResource()
BaseLLM = BaseLLM(model_name="google/flan-t5-small", load_in_8bit=False, r=32)
TrainingResource = TrainingResource(peft_model_id="lora-flan-t5-xl",
                                     num_train_epochs=1,
                                       per_device_eval_batch_size=8,
                                         per_device_train_batch_size=8,
                                         lr=1e-3)


RESOURCES = {
    'accelerator': accelerator,
    'BaseLLM': BaseLLM,
    'TrainingResource': TrainingResource
}

