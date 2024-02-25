from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse
import pandas as pd
from utils import *
from loader import *
from tqdm import tqdm
import pdb

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpu', type=str, default="0", 
                        help="Support '0', '1', 'auto'.")
    parser.add_argument('--dir_data', type=str, default="../data/")
    parser.add_argument('--dir_save', type=str, default="../result/tmp/")
    parser.add_argument('--test_data', type=str, default="test_sample.csv")
    parser.add_argument('--answer_data', type=str, default="answer.csv")
    parser.add_argument('--instruction', type=str, default="You are a helpful assistant.",
                        help="ex) Please answer the following multiple questions in Korean.")
    parser.add_argument('--max_new_tokens', type=int, default=400)
    parser.add_argument('--version_explode', default="store_true",
                        help="True if argument exist else False.")

    args = parser.parse_args()
    
    return args

def gen(x, model, tokenizer):
    input_text = f"{x}<|im_end|>\n<|im_start|>assistant\n" # Qwen
    gened = model.generate(
        **tokenizer(
            x,
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=args.max_new_tokens,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
        temperature = 0.1, # 1: 확률분포 변형 없이 사용 / temperature=100000000.0 -> 1보다 큰 값으로 두면 확률분포가 평평해지면서 기존 top-k 샘플링에서 선택되기 어려웠던 토큰들이 다음 토큰으로 선택될 수 있습니다.
        top_p = 0.1
    )
    return tokenizer.decode(gened[0])

def main(args):
    model, tokenizer = load_model_and_tokenizer(f"{args.dir_save}final/", args.num_gpu, is_train=True)

    df_test = pd.read_csv(os.path.join(args.dir_data, args.test_data))
    test = formatting_prompts_func(df_test, args.instruction, is_train=False)
    test = [tokenizer.apply_chat_template(d, tokenize=False) for d in test]
    test_data = []
    for q in tqdm(test):
        test_data.append(gen(q, model, tokenizer))
    df_test['답변'] = test_data
    df_test.to_csv(os.path.join(args.dir_save, args.answer_data), encoding='utf-8-sig', index=False)


if __name__=="__main__":
    args = define_argparser()
    main(args)