import pandas as pd
import pdb
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from utils import print_trainable_parameters, formatting_prompts_func
from datasets import Dataset, DatasetDict

def load_model_and_tokenizer(model_path, num_gpu, is_train=True):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_params = {"pretrained_model_name_or_path": model_path,
                   "quantization_config": bnb_config,
                   }
    
    token_params = {"pretrained_model_name_or_path": model_path}

    if num_gpu=="auto":
        model_params['device_map'] = "auto"
    else:
        model_params['device_map'] = {"":int(num_gpu)}
    
    if "Orion" in model_path:
        token_params['use_fast'] = False
        token_params['trust_remote_code'] = True
        model_params['trust_remote_code'] = True
        model_params['torch_dtype'] = torch.bfloat16 

    tokenizer = AutoTokenizer.from_pretrained(**token_params)
    model = AutoModelForCausalLM.from_pretrained(**model_params)

    if not is_train:
        return model, tokenizer
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    return model, tokenizer

def load_dataset(dir_data, train_data, tokenizer, instruction):
    data = pd.read_csv(os.path.join(dir_data, train_data))
    data = data[['question','answer']]

    chat_data = [tokenizer.apply_chat_template(d, tokenize=False) for d in formatting_prompts_func(data, instruction, is_train=True)]
    if tokenizer.eos_token != "</s>":
        chat_data = [f"{c}{tokenizer.eos_token}"for c in chat_data]
    #pdb.set_trace()
    chat_data = pd.DataFrame(chat_data, columns=['text'])
    chat_data = Dataset.from_dict(chat_data)

    return chat_data