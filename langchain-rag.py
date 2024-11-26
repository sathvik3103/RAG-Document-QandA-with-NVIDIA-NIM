import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv
load_dotenv()

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


llm=ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("./pdfs_folder")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("Llama3 RAG chatbot using Nvidia NIM & HuggingFace Embeddings")

prompt = ChatPromptTemplate.from_template(
    """

    Answer the questions based only on the provided.
    Please provide the moast accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """
)

prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("Embedded docements in the FAISS Vector Store are ready!")
if prompt1:
    document_chain  = create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time: ",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------")

