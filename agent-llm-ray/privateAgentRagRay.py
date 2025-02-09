import gradio as gr
from fastapi import FastAPI, File, UploadFile
from ray import serve
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import os

load_dotenv()

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class RAGDeployment:
    def __init__(self):
        self.model = os.getenv('MODEL', 'llama3.2')
        self.llm = f"ollama/{self.model}"
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.embeddings = OllamaEmbeddings(model=self.model)
        self.retriever_cache = {}

    def load_and_retrieve_docs(self, source):
        if source in self.retriever_cache:
            print("source existing = ", source)
            return self.retriever_cache[source]

        if source.startswith("http"):
            loader = WebBaseLoader(web_paths=(source,), bs_kwargs=dict())
            docs = loader.load()
        elif source.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path=source)
            docs = loader.load()
        else:
            raise ValueError("Unsupported document source.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)

        retriever = vectorstore.as_retriever()
        self.retriever_cache[source] = retriever

        return retriever

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def process_source(self, src, question, is_pdf=False):
        retriever = self.load_and_retrieve_docs(src.strip())
        retrieved_docs = retriever.invoke(question)
        formatted_context = self.format_docs(retrieved_docs)

        researcher = Agent(
            role='Researcher',
            goal='Retrieve relevant information from documents',
            backstory='You are an expert at finding and extracting relevant information from various sources.',
            allow_delegation=False,
            llm=self.llm
        )

        research_task = Task(
            description=f"Research the following question using the provided context from {'PDF' if is_pdf else 'URL'} {src}: {question}\n\nContext: {formatted_context}",
            agent=researcher,
            expected_output="An answer to the given question based on the provided context."
        )

        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        return f"{'PDF' if is_pdf else 'URL'}: {src}:\n\n{result}\n"

    @app.post("/rag_chain")
    async def rag_chain(self, source_url: str, source_pdf: list[UploadFile] = File(None), question: str = ""):
        responses = []

        if source_url:
            for src in source_url.split(","):
                responses.append(self.process_source(src.strip(), question))

        if source_pdf:
            for src in source_pdf:
                responses.append(self.process_source(src.filename, question, is_pdf=True))

        return '\n\n'.join(responses)

deployment = RAGDeployment.bind()

if __name__ == "__main__":
    serve.run(deployment)

# Gradio interface
def gradio_interface(source_url, source_pdf, question):
    import requests
    files = [('source_pdf', (file.name, file.read(), 'application/pdf')) for file in source_pdf] if source_pdf else None
    response = requests.post("http://localhost:8000/rag_chain", 
                             data={"source_url": source_url, "question": question},
                             files=files)
    return response.text

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter URL(s) separated by comma", type="text"),
        gr.File(label="Upload PDF File(s)", file_count="multiple"),
        "text"
    ],
    outputs="text",
    title="Agentic Retrieval-Augmented Generation (RAG) Chain Question Answering",
    description="Enter URL(s) separated by comma or upload a PDF file(s) to get answers from the agentic RAG chain."
)

iface.launch()
  