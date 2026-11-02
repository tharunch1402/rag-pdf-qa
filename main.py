from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path
from typing import Any,List


load_dotenv()

def get_load_document(file_path :str):

    '''  Loading a PDF File using the file path'''

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f'File not found {file_path}')

    if path.suffix.lower() != '.pdf':
        raise ValueError('Only PDF files are supported')


    loader = PyPDFLoader(str(path))
    documents = loader.load()

    return documents

def get_splitters(documents):

    ''' Splitting the doucments into chunks'''

    splitter = RecursiveCharacterTextSplitter(separators = ['\n\n','\n',' ',''],
                                               chunk_size = 500,
                                               chunk_overlap = 50)

    chunks = splitter.split_documents(documents)

    return chunks


from langchain_openai import OpenAIEmbeddings
from config import settings


def get_embedding_model():

    if not settings.OPENAI_API_KEY:
        raise ValueError('api key not found')

    embedding_model = OpenAIEmbeddings(api_key = settings.OPENAI_API_KEY,
                                       model = 'text-embedding-3-small')

    return embedding_model

from langchain_chroma import Chroma


def vector_stores(chunks):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(embedding = embedding_model,
                          documents = chunks,
                          persist_directory = './chroma_db',
                          collection_name = 'Production_RAG')

    return vector_store

if __name__ == '__main__':
    documents = get_load_document('data/sample.pdf')
    chunks = get_splitters(documents)
    vector_stores = vector_stores(chunks)


    print('Pages loaded:', len(documents))
    print('Total chunks:', len(chunks))

    # Test the search (change this to a question from YOUR PDF)
    question = 'What is this document about?'
    results = vector_stores.similarity_search(question, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} (page {doc.metadata.get('page')}) ---")
        print(doc.page_content[:300])