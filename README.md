# 🌾 CropCraft-AI

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io)

CropCraft-AI is an interactive application designed to assist users with farming and livestock-related queries. It leverages advanced AI models, speech recognition, and text-to-speech technology to provide accurate and detailed answers to user questions. The application supports multiple languages for both input and output, making it accessible to a broader audience.

## 🌟 Features

- **🎤 Voice and Text Input:** Users can ask questions either by typing or using voice commands.
- **🌐 Multilingual Support:** Answers can be translated and spoken in various languages, including English, Hindi, Telugu, and Tamil.
- **📄 Document Processing:** The application processes PDF documents to build a knowledge base for answering questions.
- **💻 Interactive UI:** A user-friendly interface built with Streamlit.

## 🛠️ Tech Stack

- **Streamlit:** For creating the web interface.
- **SpeechRecognition:** For converting speech input into text.
- **pyttsx3:** For text-to-speech conversion.
- **gTTS:** For generating speech from text in different languages.
- **GoogleTranslate:** For translating text.
- **Google Generative AI:** For generating embeddings and conversational responses.
- **FAISS:** For efficient similarity search.
- **Langchain:** For managing conversational AI models.
- **PyPDF2:** For extracting text from PDF documents.
- **dotenv:** For managing environment variables.

## 🚀 Setup

### Prerequisites

- Python 3.7 or higher
- Pip

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/CropCraft-AI.git
    cd CropCraft-AI
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add your Google API key:
        ```env
        GOOGLE_API_KEY=your_google_api_key_here
        ```

5. **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📘 Usage

1. **Upload PDF Documents:** The application processes PDF documents to build a knowledge base. Ensure that the PDF files are in the same directory as the script.
2. **Select Language:** Choose your preferred language from the dropdown menu.
3. **Ask Questions:** You can ask questions by typing in the text input box or using the voice input feature.
4. **Receive Answers:** The application will process your question and provide detailed answers. You can also listen to the answers in the selected language.

## 🤝 Contributing

We welcome contributions to enhance CropCraft-AI. Here’s how you can help:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [gTTS](https://pypi.org/project/gTTS/)
- [Google Translate](https://pypi.org/project/googletrans/)
- [Google Generative AI](https://ai.google/)
- [FAISS](https://faiss.ai/)
- [Langchain](https://langchain.ai/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

## 📬 Contact

For any inquiries or issues, please contact [your-email@example.com](mailto:your-email@example.com).

---

Feel free to explore, use, and contribute to CropCraft-AI! 🌱
