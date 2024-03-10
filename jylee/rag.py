import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from datetime import datetime, timedelta
import argparse
import time

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from peft import LoraConfig, PeftModel
import pdb
from sentence_transformers import SentenceTransformer
import logging
#from rouge import Rouge


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpu', type=str, default="auto", 
                        help="Support '0', '1', 'auto'.")
    parser.add_argument('--dir_data', type=str, default="../data/")
    parser.add_argument('--rag_data', type=str, nargs='+', default=None)
    parser.add_argument('--model_path', type=str, default="../model/orion_1/final/" ) 
    parser.add_argument('--max_length', type=int, default=5000)
    parser.add_argument('--dir_save', type=str, default="./output_rag/")
    parser.add_argument('--test_data', type=str, default="test.csv") 
    parser.add_argument('--test_sample_data', type=str, default="test_sample.csv") 
    parser.add_argument('--emb_model', type=str, default="BAAI/bge-m3")
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--add_fn', type=str, default="newdoc")

    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.1)

    args = parser.parse_args()
    
    return args

def define_logger():
    logger = logging.getLogger(name='MyLog')
    logger.setLevel(logging.INFO) ## 경고 수준 설정
    formatter = logging.Formatter('|%(asctime)s||%(levelname)s|%(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S'
                                 )
    file_handler = logging.FileHandler(os.path.join(args.dir_save, "rag_log.log")) ## 파일 핸들러 생성
    file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
    logger.addHandler(file_handler) ## 핸들러 등록

    return logger

def load_document(dir_data, rag_data):
    if "blog" in rag_data:
        loader = CSVLoader(os.path.join(dir_data, rag_data))
    else:
        loader = CSVLoader(os.path.join(dir_data, rag_data), 'contents', ['idx'])
    
    documents = loader.load()
    #documents = [i.page_content for i in documents]

    return documents

def load_model_and_tokenizer(model_path):
    # 4-bit quantization configuration
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    return model, tokenizer

def rag_answer(chain, query) :
    #prompt = f"<s>[INST] 참고한 문서의 내용을 모두 포함해서 세 문장 이상으로 답변해줘. {query} [/INST]"
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})

    return result['answer'], result["source_documents"]

def make_submission(df):
    vec_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    pred_embeddings = vec_model.encode(df['RAG답변'])
    
    submit = pd.read_csv(os.path.join(args.dir_data,'sample_submission.csv'))
    submit.iloc[:,1:] = pred_embeddings
    
    return submit

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

def cal_cos_rouge(df_test):
    vec_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    pred_embeddings = vec_model.encode(df_test['RAG답변'])
    gpt_embeddings = vec_model.encode(df_test['GPT 답변'])

    result = []
    for gpt, pred in zip(gpt_embeddings, pred_embeddings):
        result.append(cosine_similarity(gpt, pred))

    #rouge = Rouge()
    #r_score = rouge.get_scores(df_test['RAG답변'], df_test['GPT 답변'], avg=True)
    
    return result, 0


def rag(args):
    
    start_time = time.time()
    logger = define_logger()
    logger.info("")
    logger.info("Start")

    documents = None
    for rag_data in args.rag_data:
        if documents:
            new_doc = load_document(args.dir_data, rag_data)
            documents += new_doc
        else:
            documents = load_document(args.dir_data, rag_data)
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    logger.info(f"model : {model.config._name_or_path}")
    logger.info(f"model_path : {args.model_path}")
    logger.info(f"rag_doc : {args.rag_data}")

    pipe = pipeline(task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=args.max_length, 
                device_map="auto",
                torch_dtype=torch.float16,
               )
    
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={'temperature':args.temp, 
                      'top_p':args.top_p, 
                      'max_length':args.max_length
                      }
    )
        
    query_lst = pd.read_csv(os.path.join(args.dir_data, args.test_data))
    query_lst = query_lst['질문']

    embeddings = HuggingFaceEmbeddings(
        model_name=args.emb_model,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':False}
    )

    vectordb = FAISS.from_documents(
        documents = documents,
        embedding = embeddings,
    )

    # 다중질문 처리하는 prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=['question'],
        template = """
            Your task is to split multiple querys to single query that aim to answer all user's questions 
            The user questions are focused on interior, papering, and related techniques.
            Each query Must tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.
            Provide these alternative questions seperated by newlines.
            Original question : {question}
        """
    )

    retriever = vectordb.as_retriever(search_type="mmr",
                                      search_kwargs={"k":args.top_k, "fetch_k":args.top_k}) # 상위 top_k개의 결과를 반환


    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = retriever,
        llm = llm,
        prompt = QUERY_PROMPT,
    )
        
    # Chaining a pipeline
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_from_llm,
        max_tokens_limit = 4096,
        return_source_documents=True
    )

    answers_rag = []
    answers_doc = []
    for q in tqdm(query_lst) :
        a, doc = rag_answer(chain, q)
        try:
            doc = [d.page_content for d in doc] # metadata 제외하여 contents만 추출
        except:
            pdb.set_trace()
        answers_rag.append(a)
        answers_doc.append(doc)
        #time.sleep(2)

    if not os.path.isdir(args.dir_save):
        os.mkdir(args.dir_save)

    n = 1
    # 130개 답변 생성
    date_today = datetime.utcnow() + timedelta(hours=9)
    fn = f"{date_today.strftime('%Y%m%d')}_{model.config._name_or_path.split('/')[1]}_{args.add_fn}"
    while os.path.isfile(os.path.join(args.dir_save, f"{fn}_답변확인({n}).xlsx")):
        n += 1
        #if n == 100: break  
    tmp_file_name = os.path.join(args.dir_save, f"{fn}_답변확인({n}).xlsx")

    df = pd.DataFrame([list(query_lst), answers_rag, answers_doc], index = ["질문", "RAG답변","참고문서"]).T
    
    #pdb.set_trace()
    writer = pd.ExcelWriter(tmp_file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name = "130개 답변")

    # 33개 답변 생성
    df_33 = pd.read_csv(os.path.join(args.dir_data, args.test_sample_data))
    df_33 = pd.merge(df_33, df, on='질문', how='left')
    cos_33, rouge_33 = cal_cos_rouge(df_33)
    df_33['cos'] = cos_33
    df_33.to_excel(writer, sheet_name = "33개 답변")
    writer.close()
    logger.info(f"=== 파일 생성 : {tmp_file_name.split('/')[-1]}")
    
    # 제출용 파일 생성
    submit = make_submission(df)
    submit.to_csv(os.path.join(args.dir_save, f"{fn}_submission({n}).csv"), index=False)
    logger.info(f"=== 파일 생성 : {fn}_submission({n}).csv")

    logger.info(f"=== cos(33개) : {sum(cos_33)/33}")
    logger.info(f"=== rouge-l : {rouge_33}")
    logger.info(f"=== 빈 sting 개수 : {len(df[df.RAG답변.apply(lambda x:len(x.strip()))==0])}")

    end_time = time.time()
    time_result = timedelta(seconds=(end_time - start_time))
    logger.info(f"=== 총 소요 시간 : {time_result}")
    logger.info("End")

if __name__=="__main__":
    args = define_argparser()
    rag(args)