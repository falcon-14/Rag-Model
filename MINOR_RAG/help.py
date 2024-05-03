from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint
import os

# Initialize Flask app
app = Flask(__name__)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Load Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-7b")
chain = load_qa_chain(llm, chain_type="stuff")

# Load Sentence Transformer model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Pinecone index
os.environ["PINECONE_API_KEY"] = "ecfd6d7b-eada-40cb-bda5-bda66fc36395"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "agriculture"
pinecone_index = None

def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_index():
    global pinecone_index
    documents = load_docs('.')
    docs = split_docs(documents)
    pinecone_index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

@app.route('/ask', methods=['POST'])
def ask_question():
    global pinecone_index
    data = request.get_json()
    query = data['query']
    if pinecone_index is None:
        initialize_index()
    relevant_docs = get_similar_docs(query)
    response = chain.run(input_documents=relevant_docs, question=query)
    # Limit the length of the answer
    response = limit_answer_length(response)
    return jsonify({'answer': response})

def get_similar_docs(query, k=3):
    global pinecone_index
    similar_docs = pinecone_index.similarity_search(query, k=k)
    return similar_docs

def limit_answer_length(answer, max_length=450):
    if len(answer) > max_length:
        return answer[:max_length] + "..."
    else:
        return answer

@app.route('/')
def index():
    return render_template('hello.html')

if __name__ == '__main__':
    app.run(debug=True)
