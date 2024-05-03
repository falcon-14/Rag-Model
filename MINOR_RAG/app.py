from flask import Flask, render_template, request
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint
import os
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# Load Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load Pinecone API key
os.environ["PINECONE_API_KEY"] = "ecfd6d7b-eada-40cb-bda5-bda66fc36395"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Load pre-trained model for embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load question answering chain
llm = HuggingFaceEndpoint(repo_id="google/gemma-7b")
chain = load_qa_chain(llm, chain_type="stuff")

# Function to load documents from a directory
def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create Pinecone index
def create_pinecone_index(docs, embeddings, index_name="agridb"):
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=embeddings.model.get_embedding_dimension(),
            metric='cosine'
        )
    pinecone_client.index(name=index_name).upsert(vectors=embeddings.encode_documents(docs), ids=range(len(docs)))
    return pinecone_client.index(name=index_name)

# Load documents and create Pinecone index
directory = 'pest.'  # Change this to your desired directory
documents = load_docs(directory)
docs = split_docs(documents)
index = create_pinecone_index(docs, embeddings)

# Function to get similar documents
def get_similar_docs(query, k=3):
    similar_docs = index.query(queries=[embeddings.encode_query(query)], top_k=k)
    return similar_docs

# Function to get an answer to a question
def get_answer(query):
    similar_docs = get_similar_docs(query)
    response = chain.run(input_documents=similar_docs, question=query)
    return response

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    answer = get_answer(query)
    return render_template('result.html', query=query, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
