
import os
import torch
import sys
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from peft import LoraConfig, PeftModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


model_name = sys.argv[1]  # dpo_results
memo = sys.argv[2]
# model_name = 'solar_rag3'
# memo = "solar_instruct_rag"
base_model = f'../model/{model_name}/final'
dir_data = '../data/'

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

## 모델 로드
model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)

docs_dict = {os.path.join(dir_data, 'id_question_with_longest_answer_train.csv') : {'contents':'full_contents', 'meta':['id', 'full_question', 'answer']},
             os.path.join(dir_data, 'keyword_with_longest_answer.csv') : {'contents':'full_contents', 'meta':['keyword']}}

docs_nm = list(docs_dict.keys())[0]
loader = CSVLoader(docs_nm, docs_dict[docs_nm]['contents'], docs_dict[docs_nm]['meta'])
documents = loader.load()

pipe = pipeline(task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=4000,    # 텍스트의 최대 길이(텍스트의 최대 토큰 수)를 제한
                device_map="auto",
                torch_dtype=torch.bfloat16,
                batch_size=8,
               )

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={'temperature':0.2, 'max_length':4000}
)

# embedding_model = "jhgan/ko-sbert-nli"
embedding_model = "BAAI/bge-m3" # 정연전임님 임베딩 모델
model_kwargs={'device':'cuda'}
encode_kwargs = {'normalize_embeddings':False}

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = FAISS.from_documents(
    documents = documents,
    embedding = embeddings,
)

# 선택할 관련 문서 개수 정하기
retriever = vectordb.as_retriever(search_kwargs={"k":3}) # 상위 3개의 결과를 반환

# Chaining a pipeline
chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    return_source_documents=True
)

def rag_answer(query) :
    #prompt = f"<s>[INST] 모든 답변은 한국어로 한국에 맞게 설명해주세요. {query} [/INST]"
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer'], result["source_documents"]


query_lst = pd.read_csv("../data/test.csv")
query_lst = query_lst['질문']

# 테스트
# query = query_lst[0]
# chat_history = []
# chain({"question": query, "chat_history": chat_history})
from tqdm import tqdm
answers_rag = []
answers_doc = []
for q in tqdm(query_lst) :
    a, doc = rag_answer(q)
    #print(f'질문 : {q}')
    #print(f'답변 : {a}')
    answers_rag.append(a)
    answers_doc.append(doc)

today = datetime.today().strftime("%m%d")
df = pd.DataFrame([list(query_lst), answers_rag, answers_doc], index = ["질문", "답변","참고문서"]).T #.to_csv("orion_rag_m3.csv", encoding = "utf-8-sig")
df.to_csv(f"{today}_{memo}_{list(docs_dict.keys())[0].split('/')[-1].split('.')[0]}.csv", encoding = "utf-8-sig")

test = pd.read_excel("../data/dacon_llm_answer.xlsx")
test = pd.merge(test[['질문', 'GPT 답변']], df, on='질문', how='left')
preds = test['답변']
gts = test['GPT 답변']

get_cosine_similarity_score(preds, gts, f"{today}_solar_rag_answer_sample.csv")

# 스코어 계산
rouge = Rouge()
rouge.get_scores(preds, gts, avg=True)

embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

preds = df['답변']
sub_df = pd.read_csv("../data/sample_submission.csv")
pred_emds = encoding_model.encode(preds)
sub_df.iloc[:,1:] = pred_emds

sub_df.to_csv(f"../result/submission_{memo}.csv", index=False)