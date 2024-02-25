import pandas as pd
import os
from datasets import Dataset, DatasetDict
import argparse

import transformers
#from transformers import TrainingArguments
from tqdm import tqdm

#from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import *
from loader import *

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpu', type=str, default="0", 
                        help="Support '0', '1', 'auto'.")
    parser.add_argument('--dir_data', type=str, default="../data/")
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen1.5-7B-Chat") # "OrionStarAI/Orion-14B-Chat"
    parser.add_argument('--dir_save', type=str, default="../result/tmp/")
    parser.add_argument('--answer_data', type=str, default="answer.csv")
    parser.add_argument('--instruction', type=str, default="You are a helpful assistant.",
                        help="ex) Please answer the following multiple questions in Korean.")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--version_explode', default="store_true",
                        help="True if argument exist else False.")

    args = parser.parse_args()
    
    return args

def train(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from peft import LoraConfig, get_peft_model

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.num_gpu, is_train=True)

    training_args = TrainingArguments(
        output_dir=args.dir_save,          # output directory
        num_train_epochs=args.epoch,            # total number of training epochs (default : 3)
        logging_steps=500,               # How often to print logs
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,   # default(8)
        fp16=True,
        optim="paged_adamw_8bit",
        seed=3                           # Seed for experiment reproducibility 3x3
    )
    if "Qwen" in args.model_path:
        response_template = "<|im_start|>assistant\n"
    elif "Orion" in args.model_path:
        response_template = "Assistant: </s>" #</s>

    train_data = load_dataset(args.dir_data, args.train_data, tokenizer, args.instruction)
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        args=training_args,
        data_collator=collator,
        dataset_text_field='text'
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(f"{args.dir_save}final/")
    tokenizer.save_pretrained(f"{args.dir_save}final/")

if __name__=='__main__':
    args = define_argparser()
    train(args)
