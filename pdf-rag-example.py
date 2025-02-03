import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def main():
    print("Hello from pdf-rag-example!")
    st.write("this is a test app")


def pdf_rag_example():
    # Streamlit app title
    st.title("Build a RAG System with DeepSeek R1 & Ollama")

    # Load the PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Save PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        embedding_model = "sentence-transformers/all-mpnet-base-v2"

        # Load PDF text
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

        # Split text into semantic chunks
        text_splitter = SemanticChunker(
            HuggingFaceEmbeddings(model_name=embedding_model))
        documents = text_splitter.split_documents(docs)

        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.from_documents(documents, embeddings)

        # Connect retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )  # Fetch top 3 chunks

        llm = OllamaLLM(model="deepseek-r1")

        # Craft the prompt template
        prompt = """
        1. Use only the context below.
        2. If unsure, say "I don't know".
        3. Keep answers under 4 sentences.

        Context: {context}

        Question: {input}

        Helpful Answers:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        combine_docs_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Streamlit UI
        user_input = st.text_input("Ask your PDF a question")

        if user_input:
            with st.spinner("Thinking.."):
                response = rag_chain.invoke({"input": user_input})["answer"]
                st.write("Response:")
                st.write(response)

    else:
        st.write("Please upload a PDF file to begin")


if __name__ == "__main__":
    pdf_rag_example()
