import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_data', type=str, default="../data/")
    parser.add_argument('--dir_save', type=str, default="../result/")
    parser.add_argument('--test_data', type=str, default="test_sample.csv")  # chatGPT 33개 답변
    parser.add_argument('--answer_data', type=str, default="answer.csv")
    parser.add_argument('--sample_data', type=str, default="sample_submission.csv")
    parser.add_argument('--submission_data', type=str, default="submission.csv")

    args = parser.parse_args()

    return args

# 샘플에 대한 Cosine Similarity 산식
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

def get_cosine_similarity_score(encoding_model, preds, gts):
    """test data에 대한 cosine 유사도 구한 후 파일 저장하는 함수 """

    sample_scores = []
    pred_embed_lst = []
    for pred, gt in zip(preds, gts):
        # 생성된 답변 내용을 512 Embedding Vector로 변환
        pred_embed = encoding_model.encode(pred)
        gt_embed = encoding_model.encode(gt)
        pred_embed_lst.append(pred_embed)

        sample_score = cosine_similarity(gt_embed, pred_embed)
        # Cosine Similarity Score가 0보다 작으면 0으로 간주
        sample_score = max(sample_score, 0)
        sample_scores.append(sample_score)
    print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))
    return sample_scores

# 스코어 계산
def get_rouge_score(preds, gts):
    from rouge import Rouge

    rouge = Rouge()
    rouge_score = rouge.get_scores(preds, gts, avg=True)
    rouge_l_score = rouge_score['rouge-l']
    print("rouge_score:", rouge_l_score)

def save_submission_file(encoding_model, args, answer_df):

    sub_df = pd.read_csv(os.path.join(args.dir_data, args.sample_data))

    preds = answer_df['답변']
    pred_emds = encoding_model.encode(preds)
    sub_df.iloc[:,1:] = pred_emds
    sub_df.to_csv(os.path.join(args.dir_save, args.submission_data), index=False)

def main(args):
    # Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
    encoding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    test_df = pd.read_csv(os.path.join(args.dir_data, args.test_data))
    answer_df = pd.read_csv(os.path.join(args.dir_save, args.answer_data))

    # test sample  코사인 유사도, rouge 점수 구하기
    test_df = pd.merge(test_df, answer_df, on='질문', how='left')

    preds = test_df['답변']
    gts = test_df['GPT 답변']

    # test_sample cosine 유사도, rouge 점수 계산
    sample_scores = get_cosine_similarity_score(encoding_model, preds, gts)
    get_rouge_score(preds, gts)

    test_df['cosine_score'] = sample_scores
    test_df.to_csv(os.path.join(args.dir_save, args.test_data), encoding='utf-8-sig')
    # 인코딩 파일 저장
    save_submission_file(encoding_model, args, answer_df)

if __name__=="__main__":
    args = define_argparser()
    main(args)