# RAGDeployment with Ray Serve Integration 🚀

## Introduction
This project demonstrates the integration of Ray Serve for deploying a Retrieval-Augmented Generation (RAG) model. Ray Serve is used to manage the deployment of the model and handle requests from the FastAPI endpoint.

## Main Contributions 🎯

- **RAGDeployment Class**: Manages the deployment of the RAG model using Ray Serve.
- **API Integration**: Exposes the functionality via FastAPI and provides a Gradio interface for user interaction.

## Ray Serve Details 🛠️

### Overview
Ray Serve is a scalable and flexible model serving library built on top of Ray, an open-source distributed computing framework. It allows you to deploy and manage machine learning models with ease. In this project, Ray Serve is used to deploy the RAG model and handle requests from the FastAPI endpoint.

### Key Features
- **Scalability**: Ray Serve can scale horizontally to handle a large number of requests.
- **Flexibility**: It supports various machine learning frameworks and can be easily integrated with existing applications.
- **High Performance**: Ray Serve provides low-latency serving for machine learning models.

### How Ray Serve Speeds Up Performance ⚡

1. **Distributed Computing**: Ray Serve leverages the Ray framework's distributed computing capabilities. This allows the deployment to scale across multiple nodes, balancing the load and ensuring efficient resource utilization.
2. **Parallelism**: By distributing tasks across multiple workers, Ray Serve can handle numerous requests simultaneously. This parallelism reduces latency and improves throughput.
3. **Dynamic Scaling**: Ray Serve can automatically scale the number of workers based on the incoming request load. This ensures that the system can adapt to varying workloads and maintain optimal performance.
4. **Asynchronous Processing**: Ray Serve supports asynchronous processing, which means that the system can handle multiple requests concurrently without blocking. This further reduces response times and increases overall efficiency.
5. **Efficient Model Management**: Ray Serve allows for efficient management of machine learning models, including loading, updating, and versioning. This ensures that the models are always ready to handle requests with minimal overhead.

### Code Snippet

```python
from fastapi import FastAPI
from ray import serve

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
        # Load and retrieve documents based on the source
        # ...

    def process_source(self, src, question, is_pdf=False):
        # Process the source and retrieve relevant information
        # ...

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
