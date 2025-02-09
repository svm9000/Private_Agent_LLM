import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Constants
ALPHA = 0.1  # Learning rate
EPSILON = 0.2  # Clip parameter
SIMILARITY_THRESHOLD = 0.75  # Threshold for considering questions similar

# User feedback storage
feedback_repository = {}

class PPOAgent:
    def __init__(self):
        self.alpha = ALPHA  # Initial learning rate
        self.epsilon = EPSILON  # Clip parameter
        self.values = {}
        self.decay_rate = 0.99  # Learning rate decay

    def update(self, state, reward):
        if state not in self.values:
            self.values[state] = 0

        old_value = self.values[state]
        new_value = old_value + self.alpha * (reward - old_value)

        # Apply the clipping
        clipped_value = np.clip(new_value, old_value - self.epsilon, old_value + self.epsilon)

        self.values[state] = clipped_value
        self.alpha *= self.decay_rate  # Decay learning rate

    def get_value(self, state):
        return self.values.get(state, 0)

agent = PPOAgent()

class RAGDeployment:
    def __init__(self):
        self.model = os.getenv('MODEL', 'llama3.2')
        self.llm = f"ollama/{self.model}"
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.embeddings = OllamaEmbeddings(model=self.model)
        self.retriever_cache = {}

    def load_and_retrieve_docs(self, source):
        print(f"Loading and retrieving documents from source: {source}")
        try:
            if source in self.retriever_cache:
                return self.retriever_cache[source]

            if source.startswith("http") or source.startswith("https"):
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
        except ValueError as e:
            print(f"Error loading documents: {str(e)}")
            return None

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def process_source(self, src, question, is_pdf=False, is_rag=False):
        if is_rag:
            retriever = self.load_and_retrieve_docs(src.strip())
            retrieved_docs = retriever.invoke(question) if retriever else []
            formatted_context = self.format_docs(retrieved_docs)
        else:
            formatted_context = "Standard LLM context"

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

    def find_similar_question(self, question):
        if not feedback_repository:
            return None, None, 0.0

        vectorizer = TfidfVectorizer().fit_transform([question] + list(feedback_repository.keys()))
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        best_match_idx = cosine_similarities.argmax()
        best_similarity_score = cosine_similarities[best_match_idx]

        if best_similarity_score > SIMILARITY_THRESHOLD:
            best_match_question = list(feedback_repository.keys())[best_match_idx]
            best_match_response = feedback_repository[best_match_question]
            return best_match_idx, best_match_response.get('rating'), best_similarity_score

        return None, None, 0.0

    def rag_chain(self, source_url=None, question=None, source_pdf=None):
        responses = []
        response_source = "RAG"

        similar_question_response_idx, similar_response, similarity_score = self.find_similar_question(question)
        print(f"Similar response {similar_response}")
        print(f"Similarity score {similarity_score}")
        print(f"Similar question response index {similar_question_response_idx}")

        if similar_response:
            similar_question_key = list(feedback_repository.keys())[similar_question_response_idx]
            inner_response = feedback_repository[similar_question_key].get('response', {}).get('response', {})
            responses.append(inner_response)
            response_source = "Existing Repository"
        else:
            is_rag = True
            if source_url:
                for src in source_url.split(","):
                    responses.append(self.process_source(src.strip(), question, is_pdf=False, is_rag=is_rag))
                    response_source = "RAG"

            if source_pdf:
                for src in source_pdf:
                    responses.append(self.process_source(src.filename, question, is_pdf=True, is_rag=is_rag))
                    response_source = "RAG"

            state = question
            reward = agent.get_value(state)
            print(f"Reward {reward}")
            if reward < 0.5:
                is_rag = False
                response_source = "Standard LLM"

        return {'source': response_source, 'response': '\n\n'.join(responses)}

def feedback(question: str, rating: int, response: str):
    feedback_repository[question] = {'rating': rating, 'response': response}
    reward = rating / 5  # Normalize rating to 0-1
    agent.update(question, reward)
    return {"message": f"Feedback recorded: {feedback_repository[question].get('rating')} and normalized value {reward}"}

rag_deployment = RAGDeployment()

def gradio_interface(source_url, source_pdf, question, rating):
    rating_map = {
        "😃": 5,
        "😊": 4,
        "😐": 3,
        "😞": 2,
        "😡": 1
    }
    try:
        files = [('source_pdf', (file.name, file.read(), 'application/pdf')) for file in source_pdf] if source_pdf else None
        _response = rag_deployment.rag_chain(source_url=source_url, question=question, source_pdf=files)
        formatted_response = _response

        feedback_result = feedback(question, rating_map[rating], _response)
        feedback_message = feedback_result["message"]

        return formatted_response['response'] + f"\n\nSource: {formatted_response['source']}\nFeedback: {feedback_message}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter URL(s) separated by comma", type="text"),
        gr.File(label="Upload PDF File(s)", file_count="multiple"),
        gr.Textbox(label="Enter Your Question", type="text"),
        gr.Dropdown(label="Rating", choices=["😃", "😊", "😐", "😞", "😡"])
    ],
    outputs="text",
    title="Agentic Retrieval-Augmented Generation (RAG) Chain Question Answering with PPO",
    description="Enter URL(s) separated by comma or upload a PDF file(s) to get answers from the agentic RAG chain. Provide feedback on a scale of 1-5."
)

iface.launch(server_name="0.0.0.0", server_port=8000)  