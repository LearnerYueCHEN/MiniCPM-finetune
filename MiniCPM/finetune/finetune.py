# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from tqdm import tqdm

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
        user_tokens='<用户>',
        assistant_tokens='<AI>',
    ):
        super(SupervisedDataset, self).__init__()
        print("████loading json data████")
        # 逐行读取 JSON 文件，相比json.load()可以减少内存需求
        self.data = []
        self.load_data(data_path)
        
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = self.tokenizer.encode(user_tokens) #针对不同模型，都可以对应到<用户>的id
        self.assistant_tokens = self.tokenizer.encode(assistant_tokens) #针对不同模型，都可以对应到<AI>的id
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["label_ids"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def load_data(self, data_path):
        # 因为在.sh中设置的最大步数为1000，需要相应调整验证集的数据量
        if "valid" in data_path:
            max_lines = 100
            with open(data_path, 'r') as file:
                num_lines = 0
                for line in tqdm(file, total=max_lines, desc="Processing"):
                    if num_lines > max_lines-1:
                        break
                    num_lines += 1
                    # 逐行读取并解析 JSON 数据
                    try:
                        json_dict = json.loads(line.strip())
                        self.data.append(json_dict)
                    except json.JSONDecodeError as e:
                        print(f"████Error decoding JSON: {e}████")
        else:
            total_lines = sum(1 for line in open(data_path, 'r'))
            with open(data_path, 'r') as file:
                for line in tqdm(file, total=total_lines, desc="Processing"):
                    # 逐行读取并解析 JSON 数据
                    try:
                        json_dict = json.loads(line.strip())
                        self.data.append(json_dict)
                    except json.JSONDecodeError as e:
                        print(f"████Error decoding JSON: {e}████")
                    
    def preprocessing(self, example):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignore_index]

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)

            if role == "user":
                input_ids += self.user_tokens + content_ids
                label_ids += [self.ignore_index] * len(self.user_tokens) + [
                    self.ignore_index
                ] * len(content_ids)
            else:
                input_ids += self.assistant_tokens + content_ids
                label_ids += (
                    [self.ignore_index] * len(self.assistant_tokens)
                    + content_ids
                )

        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        attention_mask = [1] * len(input_ids)
        # pad to max len
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )
        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        attention_mask += [0] * (self.model_max_length - len(attention_mask))
        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = True,
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":
    model_path = "/mnt/data/user/tc_agi/yh/models/MiniCPM"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16
    )

    print("████Model and tokenizer loaded successfully.████")

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    print("████Datasets loaded successfully.████")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("████Trainer initialized successfully.████")
    
    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    model.save_pretrained("output_dir") 