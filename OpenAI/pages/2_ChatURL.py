from pickle import MEMOIZE
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import docx
from sympy import Q


# Initialize the vector database, but we need to pay openai for it
def get_vectorstore(text_chunks):
    # Create a vector store for text chunks.
    embeddings = OpenAIEmbeddings()
    # Generate embeddings for text chunks.
    vectorstore = FAISS.from_texts(texts=text_chunks  , embedding=embeddings)
    # Build vector store from text chunks and embeddings
    return vectorstore


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


def main():
    load_dotenv()
    st.header("Chat with your URL")

    url = st.text_input("Please provide your URL")

    if url is not None:
        loader = WebBaseLoader(url)
        text = loader.load()
        page_content = text[0].page_content

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=10000, chunk_overlap=200, length_function=len
        )

        # aus "page_content" chunks erstellen
        chunks = text_splitter.split_text(page_content)

        # restlicher Code bleibt unverändert, außer Text über Frage-Feld
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input(
            "Ask a question about your Website:"
        )  # Hier noch pdf in Website ändern
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
