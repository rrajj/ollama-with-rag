# TODO: add it to be history aware.


# https://python.langchain.com/docs/tutorials/qa_chat_history/
import os

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Helpful Answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def load_llm():
    llm = OllamaLLM(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, prompt, vectorstore):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given 
    language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain 
    type and configurations, and returns this QA chain. The retriever 
    is set up to return the top 3 results (k=3).

    args:
        llm : The language model to be used in the RetrievalQA.
        prompt : The prompt to be used in the chain type.
        vectorstore : The database to be used as the retriever.

    returns:
        RetrievalQA
   """
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def create_retrieval_qa_bot():

    vectorstore = Chroma(persist_directory=os.getenv('DB_PATH'), embedding_function=GPT4AllEmbeddings())

    try:
        llm = load_llm()
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


@cl.on_chat_start
async def start():
    """
    initialize bot when new chat starts
    """
    chain = create_retrieval_qa_bot()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, what is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


async def get_chain_response(chain, message_content):
    """ This function interacts with the chain and returns its response. """
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # run the bot's call method with the given message and callback.
    res = await chain.acall(message_content, callbacks=[cb])
    # print(f"response: {res}")
    return res

def process_source_documents(source_documents):
    """
    This function processes the source documents and returns a list of text elements.
    """
    text_elements = []
    for source_idx, source_doc in enumerate(source_documents):
        source_name = f"{os.path.basename(source_doc.metadata['source'])}_{source_idx}"
        
        # creating txt elements referenced in the message
        text_elements.append(
            cl.Text(content=source_doc.page_content.replace('\n', ' '), name=source_name)
        )
    source_names = [text_el.name for text_el in text_elements]
    return text_elements, source_names

@cl.on_message
async def process_chat_message(message):
    
    chain = cl.user_session.get("chain")
    res = await get_chain_response(chain, message.content)

    source_documents = res.get("source_documents", [])
    text_elements, source_names = process_source_documents(source_documents)
    
    if source_names:
        source_info = f"\nSources: {', '.join(source_names)}"
        await cl.Message(content=source_info).send()

