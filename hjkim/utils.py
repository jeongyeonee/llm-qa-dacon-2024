import os
import bitsandbytes as bnb

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def formatting_prompts_func(example, instruction, is_train=True):
    output_texts = []
    if is_train :
        for i in range(len(example['question'])):
            text = [{"role": "system", "content": instruction},
                    {"role": "user", "content": f"{example['question'][i]}"},
                    {"role": "assistant", "content": example['answer'][i]}] # assistant
            output_texts.append(text)
    else :
        for i in range(len(example['질문'])):
            text = [{"role": "system", "content": instruction},
                    {"role": "user", "content": f"{example['질문'][i]}"}]
            output_texts.append(text)

    return output_texts

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)