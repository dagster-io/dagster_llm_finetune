from dagster import Config, ConfigurableResource
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
import torch 
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

class AcceleratorResource(ConfigurableResource):
    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RawDatasetConfig(Config):
    path_to_slack_messages: str = "dataset/dagster-support-dataset.json"
    seed: int = 1
    train_size: float = 0.90
    input_text_key: str = "question"
    output_text_key: str = "gpt4_replies_target"

class BaseLLM(ConfigurableResource):
    model_name: str
    load_in_8bit: bool
    r: int = 16 
    lora_alpha: int = 32
    lora_dropout: float = .2
    bias: str = "none"
    target_modules: list = ["q", "v"]

    def PretrainedTokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def LoraModel(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, load_in_8bit=self.load_in_8bit)
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        if self.load_in_8bit:
            model = prepare_model_for_int8_training(model)    
        model = get_peft_model(model, lora_config)
        return model 


class TrainingResource(ConfigurableResource):
    peft_model_id: str = "lora-flan-t5-base"
    num_train_epochs: int = 1
    per_device_eval_batch_size: int = 8
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    def training_args(self): 
        training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=self.per_device_eval_batch_size,
        per_device_train_batch_size=self.per_device_train_batch_size,
        gradient_accumulation_steps=self.gradient_accumulation_steps,
        output_dir=self.peft_model_id,
        auto_find_batch_size=True,
        learning_rate=self.lr,
        num_train_epochs=self.num_train_epochs,
        logging_dir=f"{self.peft_model_id}/logs",
        use_mps_device=False
    )
        return training_args