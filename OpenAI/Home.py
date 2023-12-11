from pickle import MEMOIZE
import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
import requests
import openai
from sympy import Q
from htmlTemplates import css, bot_template, user_template


# Get the PDF Text Method
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Get the Chunks from the raw text of multiple PDFS
def get_text_chunks(raw_text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


# Initialize the vector database, but we need to pay openai for it
def get_vectorstore(text_chunks):
    # Create a vector store for text chunks.
    embeddings = OpenAIEmbeddings()
    # Generate embeddings for text chunks.
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # Build vector store from text chunks and embeddings
    return vectorstore

# # Vector storing but for free
# def get_vectorstore_huggingface(text_chunks):
#     # Create a vector store for text chunks.
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     # Generate embeddings for text chunks.
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     # Build vector store from text chunks and embeddings
#     return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # First Argument the Chat Model
        llm=llm,
        # 2nd Argument the Vector database
        retriever=vectorstore.as_retriever(),
        # The memory just initialized
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        # Array in denen die Konverstaion geladen wird 2 immer der Mensch, 1 der Bot.
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF´s here and cick on Process", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Process"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # helper for seeing the raw text of pdf docs
                st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # Create Conversation Chain
                # Session_State prevents Streamlit to re-load the whole code which would result in losing the memory of the AI
                st.session_state.conversation = get_conversation_chain(vectorstore)

    # # extract the text
    # if pdf is not None:
    #     pdf_reader = PdfReader(pdf)
    #     text = ""
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()

    #     # split into chunks
    #     text_splitter = CharacterTextSplitter(
    #         separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    #     )

    #     chunks = text_splitter.split_text(text)

    #     # create embeddings
    #     embeddings = OpenAIEmbeddings()
    #     knowledge_base = FAISS.from_texts(chunks, embeddings)

    #     # show user input
    #     user_question = st.text_input("Ask a question about your PDF:")

    #     # show user input
    #     if user_question:
    #         docs = knowledge_base.similarity_search(user_question)
    #         llm = OpenAI()
    #         chain = load_qa_chain(llm, chain_type="stuff")
    #         with get_openai_callback() as cb:
    #             response = chain.run(input_documents=docs, question=user_question)
    #             print(cb)
    #         st.write(response)

    # url = st.text_input("Please provide your URL")

    # if url is not None:
    #     loader = WebBaseLoader(url)
    #     text = loader.load()
    #     page_content = text[0].page_content

    #     text_splitter = CharacterTextSplitter(
    #         separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    #     )

    #     # aus "page_content" chunks erstellen
    #     chunks = text_splitter.split_text(page_content)

    #     # restlicher Code bleibt unverändert, außer Text über Frage-Feld
    #     # create embeddings
    #     embeddings = OpenAIEmbeddings()
    #     knowledge_base = FAISS.from_texts(chunks, embeddings)

    #     # show user input
    #     user_question = st.text_input(
    #         "Ask a question about your Website:"
    #     )  # Hier noch pdf in Website ändern
    #     if user_question:
    #         docs = knowledge_base.similarity_search(user_question)

    #         llm = OpenAI()
    #         chain = load_qa_chain(llm, chain_type="stuff")
    #         with get_openai_callback() as cb:
    #             response = chain.run(input_documents=docs, question=user_question)
    #             print(cb)
    #         st.write(response)


if __name__ == "__main__":
    main()
