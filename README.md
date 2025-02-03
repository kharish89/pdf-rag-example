# PDF RAG System with DeepSeek & Ollama

A Streamlit-based application that implements a Retrieval-Augmented Generation (RAG) system for querying PDF documents using DeepSeek R1 model and Ollama.

## Features

- PDF document upload and processing
- Semantic text chunking for better context understanding
- RAG implementation using FAISS vector store
- Interactive Q&A interface
- Powered by DeepSeek R1 model through Ollama
- Embeddings using HuggingFace's sentence-transformers

## Prerequisites

- Python 3.8+
- Ollama installed locally with DeepSeek R1 model
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kharish89/pdf-rag-example.git
cd pdf-rag-example
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install streamlit langchain langchain-community langchain-experimental faiss-cpu pdfplumber langchain-ollama sentence-transformers
```

4. Install and run Ollama with DeepSeek R1:
```bash
# First install Ollama from: https://ollama.ai/
ollama pull deepseek-r1
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run pdf-rag-example.py
```

2. Use the application:
   - Upload a PDF file using the file uploader
   - Wait for the document to be processed
   - Type your questions in the text input field
   - Get AI-generated responses based on the PDF content

## How It Works

1. **Document Processing**:
   - Uploads PDF documents
   - Splits text into semantic chunks using LangChain's SemanticChunker
   - Generates embeddings using HuggingFace's sentence-transformers

2. **RAG System**:
   - Creates a FAISS vector store for efficient similarity search
   - Implements a retrieval chain to fetch relevant context
   - Uses DeepSeek R1 model for generating responses

3. **Response Generation**:
   - Retrieves top 3 most relevant chunks for each query
   - Uses a custom prompt template for consistent and concise answers
   - Limits responses to 4 sentences for clarity

## Technical Components

- **Frontend**: Streamlit
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **LLM**: DeepSeek R1 via Ollama
- **Vector Store**: FAISS
- **Document Processing**: PDFPlumber
- **Framework**: LangChain

## Limitations

- Requires local installation of Ollama
- Processing large PDFs may take time
- Responses are limited to 4 sentences
- Quality of answers depends on PDF content and clarity

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG implementation framework
- Ollama for local LLM hosting
- HuggingFace for the embedding models
- Streamlit for the web interface

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/pdf-rag-example

