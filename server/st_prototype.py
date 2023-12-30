import streamlit as st
import os
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import shutil
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from utils import KaggleCompetitionDownloader

JUPYTER_NB_CACHE_DIR = Path(".cache") / "jupyter_nb"
EMBEDDING_CACHE_DIR = Path(".cache") / "embeddings"
PROMPT_TEMPLATE = """Answer the question based on the following context:
    {context}

    Question: {question}
    """


def available_competitions():
    return os.listdir(str(JUPYTER_NB_CACHE_DIR))


def load_document(path):
    doc = UnstructuredMarkdownLoader(path).load()
    split_doc = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(doc)
    return split_doc


def format_docs(docs):
    return "\n\n----------------------------------------------------\n\n".join([d.page_content for d in docs])


@st.cache_resource
def create_vector_store(directory, api_key):
    documents = os.listdir(str(directory))
    split_docs = []
    for doc in documents:
        split_docs.extend(load_document(os.path.join(directory, doc)))
    underlying_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )
    vectorstore = FAISS.from_documents(split_docs, embedding=cached_embedder)
    return split_docs, vectorstore


def create_chain(competition_name, api_key, model_name="gpt-3.5-turbo-1106"):
    # create retriever
    split_docs, vectorstore = create_vector_store(
        str(JUPYTER_NB_CACHE_DIR / competition_name), api_key=api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    )
    # create chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI(model=model_name, openai_api_key=api_key, streaming=True)
    return (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

def generate_response(input_text, api_key, model_name):
    chain = create_chain(competition, api_key, model_name)
    with st.chat_message("Bot:") :
        message_placeholder = st.empty()
        full_text = ""
        for chunk in chain.stream(input_text):
            full_text += chunk
            message_placeholder.markdown(full_text + "▌")
        message_placeholder.markdown(full_text)
    return chain

competitions = available_competitions()

st.title("Chat with Kaggle competitions")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.markdown("# Select a model")
model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
st.sidebar.markdown("# Select a competition")
competition = st.sidebar.selectbox("Competition", competitions)

st.sidebar.markdown("# Add a competition")
with st.sidebar.form("add_competition"):
    competition_to_add = st.text_input("Competition name")
    competition_added = st.form_submit_button("Add")
    dir_path = JUPYTER_NB_CACHE_DIR / competition_to_add
    nb_path = JUPYTER_NB_CACHE_DIR / (competition_to_add + "ipynb")
    if competition_added and not competition_to_add:
        st.warning("Please enter a competition name!")
    if competition_added and competition_to_add and os.path.exists(dir_path):
        st.warning("Competition already added!")
    if competition_added and competition_to_add and not os.path.exists(dir_path):
        progress_text = "Create directories..."
        my_bar = st.progress(0, text=progress_text)
        # create top level directory
        os.mkdir(dir_path)
        # create jupyter notebooks directory
        os.mkdir(nb_path)
        my_bar.progress(15, text="Downloading notebooks...")
        try:
            downloader = KaggleCompetitionDownloader(competition_to_add)
            print(nb_path)
            print(type(nb_path))
            downloader.download_all_kernels(nb_path)
        except ValueError:
            st.error("No kernels found for this competition!")
            os.rmdir(dir_path)
            os.rmdir(nb_path)
            competition_to_add = ""
            my_bar.empty()
        else:
            my_bar.progress(30, text="Converting notebooks...")
            downloader.convert_all_kernels(nb_path, dir_path)
            shutil.rmtree(nb_path)
            my_bar.progress(95, text="Cleaning up...")
            st.success("Competition added!")
            my_bar.empty()


used_chain = None

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What is this competition about?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if not competition:
        st.warning("Please select a competition!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        used_chain = generate_response(text, openai_api_key, model_name)

# write context

if used_chain is not None:
    st.markdown("## Context")
    st.markdown(used_chain.steps[0].invoke(text)['context'])
