import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import ollama
import os

# Load environment variables
load_dotenv()
# Set global model variables
model = os.getenv('MODEL', 'llama3.2')
llm =  "ollama/" + model 
chunk_size = int(os.getenv('CHUNK_SIZE', 500))
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))

embeddings = OllamaEmbeddings(model=model)
retriever_cache = {}

# Function to load, split, and retrieve documents from URL or PDF
def load_and_retrieve_docs(source):
    if source in retriever_cache:
        print("source existing = ", source)
        return retriever_cache[source]

    if source.startswith("http"):
        loader = WebBaseLoader(web_paths=(source,), bs_kwargs=dict())
        docs = loader.load()
    elif source.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path=source)
        docs = loader.load()
    else:
        raise ValueError("Unsupported document source.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    retriever_cache[source] = retriever

    return retriever

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain for multiple URLs or PDFs using CrewAI
def rag_chain(source_url, source_pdf, question):
    print("source_url = ", source_url)
    print("source_pdf = ", source_pdf)
    print("retriever_cache = ", retriever_cache)

    responses = []

    # Define CrewAI agents
    researcher = Agent(
        role='Researcher',
        goal='Retrieve relevant information from documents',
        backstory='You are an expert at finding and extracting relevant information from various sources.',
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role='Writer',
        goal='Compose comprehensive and accurate answers based on the retrieved information',
        backstory='You are a skilled writer capable of synthesizing information into clear and concise answers.',
        allow_delegation=False,
        llm=llm

    )

    # Process URLs
    if source_url:
        for src in source_url.split(","):
            retriever = load_and_retrieve_docs(src.strip())
            retrieved_docs = retriever.invoke(question)
            formatted_context = format_docs(retrieved_docs)

            # Define tasks for CrewAI
            research_task = Task(
                description=f"Research the following question using the provided context from URL {src}: {question}\n\nContext: {formatted_context}",
                agent=researcher,
                expected_output="An answer to the given question based on the provided context."
            )

            writing_task = Task(
                description="Answer the question using the provided context",
                agent=writer,
                expected_output="An answer to the given question based on the provided context."
            )

            # Create and run the crew
            crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                process=Process.sequential
            )

            result = crew.kickoff()
            response_text = f"URL: {src}:\n\n{result}\n"
            responses.append(response_text)

    # Process PDFs
    if source_pdf:
        for src in source_pdf:
            retriever = load_and_retrieve_docs(src.strip())
            retrieved_docs = retriever.invoke(question)
            formatted_context = format_docs(retrieved_docs)

            # Define tasks for CrewAI
            research_task = Task(
                description=f"Research the following question using the provided context from PDF {src}: {question}\n\nContext: {formatted_context}",
                agent=researcher,
                expected_output="An answer to the given question based on the provided context."
            )

            writing_task = Task(
                description="Answer the question using the provided context",
                agent=writer,
                expected_output="An answer to the given question based on the provided context."
            )

            # Create and run the crew
            crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                process=Process.sequential
            )

            result = crew.kickoff()
            response_text = f"PDF: {src}:\n\n{result}\n"
            responses.append(response_text)

    return '\n\n'.join(responses)

# Gradio interface setup with file upload or URL input for PDFs
iface = gr.Interface(
    fn=rag_chain,
    inputs=[
        gr.Textbox(label="Enter URL(s) separated by comma", type="text"),
        gr.File(label="Upload PDF File(s)", file_count="multiple"),
        "text"
    ],
    outputs="text",
    title="Agentic Retrieval-Augmented Generation (RAG) Chain Question Answering",
    description="Enter URL(s) separated by comma or upload a PDF file(s) to get answers from the agentic RAG chain."
)

# Launch the Gradio app
iface.launch()
