from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_KEY = os.getenv("API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["API_KEY"] = API_KEY

# --- Embeddings + Pinecone ---
embeddings = download_embeddings()
index_name = "ai-medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# --- LLM ---
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=API_KEY
)

# --- Memory (last 5 messages) ---
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,
    return_messages=True
)

# --- Prompt with memory + context ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_1),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Relevant context: {context}")
])

# --- Retrieval + memory aware chain ---
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    # Run RAG chain with both input + memory
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    # Save this exchange into memory
    memory.save_context({"input": msg}, {"output": response["answer"]})
    return str(response["answer"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port, debug=True)
