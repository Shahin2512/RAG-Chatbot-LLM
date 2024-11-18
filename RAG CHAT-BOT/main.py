from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq 
import os

from htmltemplate import css, bot_template, user_template


custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# using HuggingFace embeddings model and Chroma to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)  # Using Chroma
    return vectorstore

# generating conversation chain
def get_conversationchain(vectorstore):
    # Initialize Groq model instead of OpenAI
    groq_api_key = os.getenv("GROQ_API_KEY")  # Make sure your Groq API key is set in the environment variables
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",  # Replace with your preferred Groq model
        temperature=0,
        groq_api_key=groq_api_key,
    )
    
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')  # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content,), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # Get the pdf text
                raw_text = get_pdf_text(docs)

                # Get the text chunks
                text_chunks = get_chunks(raw_text)

                # Create vectorstore using Chroma
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversationchain(vectorstore)


if __name__ == '__main__':
    main()
