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

if __name__ == '__main__':
    documents = get_load_document('data/sample.pdf')
    chunks = get_splitters(documents)

    print('Pages loaded:', len(documents))
    print('Total chunks:', len(chunks))
    print('\n--- Chunk 1 ---\n', chunks[0].page_content)
    print('\nMetadata:', chunks[0].metadata)




