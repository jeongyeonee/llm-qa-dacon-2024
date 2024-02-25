import torch
import pdb
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def run_rag():
    loader = CSVLoader(file_path="../data/train_final_0216_onecol.csv",encoding='utf-8')
    data = loader.load()

    modelPath = "distiluse-base-multilingual-cased-v1"
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.load_local("../faiss_index", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    model_id = "../model/orion_1/final/"
                                            
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True,torch_dtype = torch.bfloat16 )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    hf = HuggingFacePipeline(pipeline=pipe)

    template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

    {context}

    질문: {question}

    유용한 답변:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    #pdb.set_trace()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | hf
        | StrOutputParser()
    )
    pdb.set_trace()
    for chunk in rag_chain.stream("도배지에 녹은 자국이 발생하는 주된 원인과 그 해결 방법은 무엇인가요?"):
        print(chunk, end="", flush=True)

if __name__== '__main__':
    run_rag()
