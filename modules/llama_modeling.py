# Major change since July 9 for scaling up
# Major change since July 18 for fixing the lora bug
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from peft import (get_peft_model, LoraConfig)
from torch.nn.functional import gelu
from transformers import BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat-hf")
    memory_head: bool = field(default=False, metadata={"help": "whether to add a memory head for the encoder."})
    better_transformer: bool = field(default=False,
                                     metadata={"help": "whether to enable bettertransformer for flash attention."})
    mem_size: int = field(default=128, metadata={"help": "Memory size"}, )
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    quantization: bool = field(default=False, metadata={"help": "quantization"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False,
                             metadata={"help": "Enable debug dataset to quickly verify the training process"})


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}, )
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Per device train batch size."}, )
    bf16: bool = field(default=False, metadata={"help": "Use brain float 16"}, )
    lm_ratio: float = field(default=0.0, metadata={"help": "Ratio for LM training."}, )
    restore_from: str = field(default="",
                              metadata={"help": "The checkpoint that should be restored from for fine-tuning"})


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: "
          f"{100 * trainable_parameters / all_param}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


class MemoryHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense_in = nn.Linear(dim, dim)
        self.dense_out = nn.Linear(dim, dim)

    def forward(self, x):
        previous_type = x.dtype
        x = x.to(self.dense_in.weight.dtype)
        x = self.dense_in(x)
        x = gelu(x)
        return self.dense_out(x).to(previous_type)


class LlamaLora(nn.Module):
    def __init__(self, model_args, training_args, gofa_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        # self.auto_encoder = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.quantization = model_args.quantization
        self.icae = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16) # [PAD] token

        self.eos_id = 1
        self.dim = self.icae.config.hidden_size
        # if self.quantization:
        #     self.icae = prepare_model_for_kbit_training(self.icae)
        lora_config = self.create_lora_config()
        self.icae = get_peft_model(self.icae, lora_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.left_tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.left_tokenizer.padding_side = "left"
        self.left_tokenizer.truncation_side = "left"

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        return bnb_config

    def create_lora_config(self):
        lora_config = LoraConfig(

            r=self.model_args.lora_r,

            lora_alpha=32,

            lora_dropout=self.model_args.lora_dropout,

            bias="none",

            task_type="CAUSAL_LM"

        )
        return lora_config

    def get_tokenizer(self):
        return self.tokenizer
