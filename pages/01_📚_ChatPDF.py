from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import streamlit as st


loader = PyPDFLoader(r'C:\Users\rianb\PycharmProjects\LLAMA2_ChatBot\data\The-Field-Guide-to-Data-Science.pdf')

data = loader.load()

print(data)