from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings()
vector_store = Chroma(persist_directory ="./chroma_db", embedding_function=embedding_function)


def load_and_split_document(file_path):
    loader = PyMuPDFLoader(file_path)
    document = loader.load()
    return text_splitter.split_documents(document)


def index_document_to_chroma(file_path, file_id):
    try:
        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata['file_id'] = file_id

        vector_store.add_documents(splits)

        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False
    

def delete_doc_from_chroma(file_id):
    try:
        docs = vector_store.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vector_store._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False