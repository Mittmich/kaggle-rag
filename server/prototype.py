#!/usr/bin/env python
from operator import itemgetter
from fastapi import FastAPI
import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

# 1. Chain definition

def load_document(path):
    doc = UnstructuredMarkdownLoader(path).load()
    split_doc = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(doc)
    return split_doc

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def create_vector_store(directory):
    documents = os.listdir(directory)
    split_docs = []
    for doc in documents:
        split_docs.extend(load_document(os.path.join(directory, doc)))
    underlying_embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )
    vectorstore = FAISS.from_documents(
        split_docs, embedding=cached_embedder
    )
    return split_docs, vectorstore

#  create vector store for dicussions


split_docs, vectorstore = create_vector_store(r'C:\Users\michael.mitter\Documents\Programming\kaggle-rag\input_data\kaggle_v1')

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)

# create vector store for notebooks

split_docs_nb, vectorstore_nb = create_vector_store(r'C:\Users\michael.mitter\Documents\Programming\kaggle-rag\input_data\kernels_md')

retriever_nb= vectorstore_nb.as_retriever(search_kwargs={"k": 3})

bm25_retriever_nb = BM25Retriever.from_documents(split_docs_nb)
bm25_retriever_nb.k = 3

ensemble_retriever_nb = EnsembleRetriever(
    retrievers=[bm25_retriever_nb, retriever_nb], weights=[0.5, 0.5]
)


# create chain discussion

template = """Answer the question based on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


model = ChatOpenAI(model="gpt-3.5-turbo-1106")
#model = ChatOpenAI(model="gpt-4-1106-preview")
chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# create chain notebooks

template = """Answer the question based on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


#model = ChatOpenAI(model="gpt-3.5-turbo-1106")
model = ChatOpenAI(model="gpt-4-1106-preview")
chain_nb = (
    {"context": ensemble_retriever_nb | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# 2. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    chain,
    path="/discussion",
)

add_routes(
    app,
    chain_nb,
    path="/notebooks",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)