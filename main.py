import pandas as pd

import os
from dotenv import load_dotenv

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import streamlit as st

from pprint import pprint


def load_dataset(dataset_name:str="dataset.csv") -> pd.DataFrame:
    """
    Load dataset from file_path

    Args:
        dataset_name (str, optional): Dataset name. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: Dataset
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df

def create_chunks(dataset:pd.DataFrame, chunk_size:int, chunk_overlap:int) -> list:
    """
    Create chunks from the dataset

    Args:
        dataset (pd.DataFrame): Dataset
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap

    Returns:
        list: List of chunks
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )
    # aggiungiamo i metadati ai chunk stessi per facilitare il lavoro di recupero
    for doc in text_chunks:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
        doc.page_content = final_content

    return text_chunks

def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get vector store

    Args:
        chunks (list): List of chunks

    Returns:
        FAISS: Vector store
    """
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings() # if you want to use open source embeddings

    if not os.path.exists("./db"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(
            chunks, embeddings
        )
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore

def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = ChatOpenAI(model="gpt-4")
    # llm = HuggingFaceHub(model="HuggingFaceH4/zephyr-7b-beta") # if you want to use open source LLMs
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit

    Args:
        user_question (str): User question
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"
    chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(
                f"<p style='text-align: right;'><b>User</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )

def main():
    load_dotenv()
    df = load_dataset()
    chunks = create_chunks(df, 1000, 0)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

        You should never answer a question with a question, and you should always respond with the most relevant documentation page.

        Do not answer questions that are not about the LangChain library or project.

        Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
        {context}
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":books:",
    )

    st.title("Documentation Chatbot")
    st.subheader("Chat with LangChain's documentation!")
    st.markdown(
        """
        This chatbot was created to answer questions about the LangChain project documentation.
        Ask a question and the chatbot will respond with the most relevant page of the documentation.
        """
    )
    st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e") # Image rights to Alex Knight on Unsplash

    user_question = st.text_input("Ask your question")
    with st.spinner("Processing..."):
        if user_question:
            handle_style_and_responses(user_question)

    st.session_state.conversation = get_conversation_chain(
        st.session_state.vector_store, system_message_prompt, human_message_prompt
    )


if __name__ == "__main__":
    main()
