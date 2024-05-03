from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

# Set Hugging Face API token (replace with your own)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load Gemma model
llm = HuggingFaceEndpoint(repo_id="google/gemma-7b")
chain = load_qa_chain(llm, chain_type="stuff")

# Load Sentence Transformer model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Pinecone index
os.environ["PINECONE_API_KEY"] = "ecfd6d7b-eada-40cb-bda5-bda66fc36395"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "agriculture"  # You can change the index name
pinecone_index = None

# Function to load documents
def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to initialize Pinecone index
def initialize_index():
    global pinecone_index
    documents = load_docs('.')  # Load documents from the current directory
    docs = split_docs(documents)
    pinecone_index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Function to create the Gemma conversational chain
def get_conversational_chain_gemma():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say "Answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = HuggingFaceEndpoint(repo_id="google/gemma-7b")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Create the Gemma conversational chain
conversational_chain = get_conversational_chain_gemma()

# Route to handle questions
@app.route('/ask', methods=['POST'])
def ask_question():
    global pinecone_index
    data = request.get_json()
    query = data['query']

    if pinecone_index is None:
        initialize_index()

    relevant_docs = get_similar_docs(query)
    response = conversational_chain.run(input_documents=relevant_docs, question=query)

    # Limit the length of the answer
    response = limit_answer_length(response)
    return jsonify({'answer': response})

# Function to get similar documents from Pinecone
def get_similar_docs(query, k=3):
    global pinecone_index
    similar_docs = pinecone_index.similarity_search(query, k=k)
    return similar_docs

# Function to limit answer length
def limit_answer_length(answer, max_length=350):
    if len(answer) > max_length:
        return answer[:max_length] + "..."
    else:
        return answer

# Index route (you'll need to create a 'hello.html' template)
@app.route('/')
def index():
    return render_template('hello.html')

if __name__ == '__main__':
    app.run(debug=True)