from dotenv import load_dotenv
load_dotenv()
import os
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI

loader = PyPDFDirectoryLoader(os.getenv('DATA_PATH'))
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

vectorStore = Chroma.from_documents(
        documents=splits,
        embedding=GPT4AllEmbeddings(),
        persist_directory=os.getenv('DB_PATH')
    )

print("DONE")