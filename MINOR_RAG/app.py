import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from googletrans import Translator
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "Your_API_Token"  

st.set_page_config(page_title="CropCraft-AI", layout="wide")

engine = pyttsx3.init()

# Function to check if the vector store exists
def vector_store_exists():
    return os.path.exists("faiss_index")

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer: 
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to get user question from speech
def get_audio_question():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say your question:")
        audio = r.listen(source)
    try:
        user_question = r.recognize_google(audio)
        st.write(f"You said: {user_question}")
        return user_question
    except:
        st.error("Sorry, I couldn't understand your question.")
        return None

def speak_answers(answers, lang='te'):  
    translator = Translator()
    for i, answer in enumerate(answers):
        translated_answer = translator.translate(answer, dest=lang).text
        tts = gTTS(text=translated_answer, lang=lang)
        filename = f'answer{i}.mp3'
        tts.save(filename)
        st.audio(filename)  

def main():
    st.header("Crop Craft AI")

    # Add a dropdown menu for language selection
    lang = st.selectbox('Select a language:', ('English', 'Hindi', 'Telugu', 'Tamil'))
    lang_code = 'en' if lang == 'English' else 'hi' if lang == 'Hindi' else 'ta' if lang=='Tamil' else 'te'

    if not vector_store_exists():
        with st.spinner("Processing documents..."):
            # Get list of PDF files from directory
            pdf_dir = "."  
            pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            raw_text = '.'
            for pdf_file in pdf_files:
                raw_text += get_pdf_text(pdf_file)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Documents processed!")
    else:
        st.info("Vector store loaded!")

    user_questions = []  # Collect user questions
    answers = []        # Collect answers

    st.button("Ask Question with Text")
    if st.button("Ask Question with Voice"):
        user_question = get_audio_question()
        if user_question:
            user_questions.append(user_question)
    else:
        user_question = st.text_input("Please Ask Your Queries Regarding Agriculture and Livestock", key="user_question")
        if user_question:
            user_questions.append(user_question)

    # Process questions and get answers
    try:
        # Process questions and get answers
        for question in user_questions:
            answer = user_input(question)
            answers.append(answer)
            st.write(answer)

        speak_answers(answers, lang=lang_code)
    except RuntimeError as e:
        st.error(f"An error occurred: {str(e)}")
        engine.stop() 

if __name__ == "__main__":
    main()
