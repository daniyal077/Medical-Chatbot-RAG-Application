# Medical Chatbot (RAG Application with Gen AI)

This repository contains a **Retrieval-Augmented Generation (RAG) medical chatbot** using **HuggingFace models**, **FAISS**, and **Streamlit**. The chatbot provides accurate medical responses based on provided documents.

## Features
- **Retrieval-Augmented Generation (RAG)**: Uses FAISS for document retrieval and a HuggingFace model for response generation.
- **HuggingFace LLM Integration**: Utilizes `Mistral-7B-Instruct-v0.3` for generating medical responses.
- **FAISS Vector Store**: Stores embeddings for efficient document retrieval.
- **Streamlit Web Interface**: Provides an interactive UI for querying the chatbot.
- **PDF Document Processing**: Loads and processes PDFs using `PyPDFLoader`.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/daniyal077/Medical-Chatbot-RAG-Application.git
   cd Medical-Chatbot-RAG-Application
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```sh
   echo "HF_TOKEN=your_huggingface_api_key" > .env
   ```

## Usage

### 1. Run the Medical Chatbot (Streamlit UI)
```sh
streamlit run main.py
```

## File Structure
```
📂 medical-chatbot
├── 📂 data                    # Folder to store medical PDFs
├── 📂 vectorstore             # Folder to store FAISS index
├── create_database.py         # Script to create FAISS database from PDFs
├── create_database_with_llm.py # Terminal-based chatbot script
├── main.py                    # Streamlit chatbot interface
├── requirements.txt            # Required Python libraries
├── .env                        # HuggingFace API key
└── README.md                   # Project documentation
```

## Dependencies
- Python 3.8+
- `langchain`
- `streamlit`
- `faiss-cpu`
- `huggingface_hub`
- `sentence-transformers`

## Future Improvements
- Add support for **multi-document retrieval**
- Improve **response generation accuracy**
- Deploy on **cloud platforms**

## License
This project is open-source and available under the MIT License.

