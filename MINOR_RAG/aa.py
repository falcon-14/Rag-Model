import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pinecone import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint

from huggingface_hub import login
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint
from pinecone import Pinecone as PineconeClient #Importing the Pinecone class from the pinecone package
from langchain_community.vectorstores import Pinecone

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "ecfd6d7b-eada-40cb-bda5-bda66fc36395"
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
index_name="agridb"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(repo_id="google/gemma-7b")
chain = load_qa_chain(llm, chain_type="stuff")

def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs('.')
len(documents)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def get_similiar_docs(query, k=3):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(query):
  relevant_docs = get_similiar_docs(query)
  response = chain.run(input_documents=relevant_docs, question=query)
  return response

css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/Wg3Gw4N/sapling-removebg-preview.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/2qrvHx8/user-removebg-preview.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""

st.title("Question Answering App")
our_query = st.text_input("Enter your question:")
if our_query:
    answer = get_answer(our_query)
    st.markdown(user_template.replace("{{MSG}}", our_query), unsafe_allow_html=True)
    st.markdown(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
