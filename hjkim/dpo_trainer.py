import os
import torch
import random

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import print_trainable_parameters, find_all_linear_names

output_dir="../model/dpo_solar_small/"
model_name = "../model/solar_rag3/final" #orion
# model_name = 'Upstage/SOLAR-10.7B-Instruct-v1.0'

# dataset = load_dataset("Intel/orca_dpo_pairs")
from datasets import load_dataset, Dataset
dataset_all = load_dataset("json", data_files="../data/dpo_dataset_kor_eng_0302.json", split='train')
# 랜덤한 인덱스 생성
random_indices = random.sample(range(len(dataset_all)), 100)

# 랜덤하게 추출된 샘플 출력
dataset = Dataset.from_list([dataset_all[idx] for idx in random_indices])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name
                                            , torch_dtype=torch.bfloat16
                                            , quantization_config=bnb_config
                                            , trust_remote_code=True
                                            , use_cache=False
                                            # , device_map='auto'
                                            # , device_map={'':torch.cuda.current_device()}
                                            )

# model = torch.nn.DataParallel(model, device_ids=[0,1]) # GPU 0,1,2,3 총 4개 사용
# model.cuda()

# model.to('cuda')

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# torch.cuda.set_device(torch.device(f'cuda:1'))
# model_ref = AutoModelForCausalLM.from_pretrained(model_name
#                                             , torch_dtype=torch.bfloat16
#                                             , quantization_config=bnb_config
#                                             , trust_remote_code=True
#                                             , use_cache=False
#                                             , device_map='auto'
#                                             # , device_map={'':torch.cuda.current_device()}
#                                             )

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=1,
    # save_steps= 300,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=100,
    output_dir=output_dir,
    optim="paged_adamw_8bit", #"paged_adamw_32bit",
    # lr_scheduler_type="cosine",
    # warmup_ratio=0.05,
    remove_unused_columns=False
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=8,
    target_modules=["v_proj","q_proj"],  #find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # None,  #model_ref
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_length=2048,
)

dpo_trainer.train()
dpo_trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)