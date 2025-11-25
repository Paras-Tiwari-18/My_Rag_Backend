import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
HF_API = os.getenv("hugging_face_api")     
GROQ_API = os.getenv("GROQ_API_KEY")

if not GROQ_API:
    raise ValueError("GROQ_API_KEY not set in .env")


BOOKS_DIR = "books"   

if not os.path.exists(BOOKS_DIR):
    raise FileNotFoundError(
        f"Books folder '{BOOKS_DIR}' not found. Create it and add Tagore PDFs inside."
    )

all_docs = []

for fname in os.listdir(BOOKS_DIR):
    if fname.lower().endswith(".pdf"):
        pdf_path = os.path.join(BOOKS_DIR, fname)
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        book_name = os.path.splitext(fname)[0]
        for d in docs:
            d.metadata["source_book"] = book_name
        all_docs.extend(docs)

if not all_docs:
    raise ValueError(
        f"No PDF files found in {BOOKS_DIR}. Please add Tagore PDFs there."
    )
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
final_documents = text_splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(final_documents, embeddings)
retriever = vectorstore.as_retriever()

retriever.search_kwargs["k"] = 10



llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=GROQ_API)


emotion_system_prompt = """
You are an emotion classifier for a counseling chatbot.
Given the user's message, return ONLY ONE label from this list:

["sadness", "anxiety", "heartbreak", "loneliness", "confusion",
 "stress", "burnout", "self-doubt", "anger", "mixed", "neutral"]

Rules:
- Respond with JUST the label, nothing else.
- If it's hard to guess, use "mixed".
- If the user is casual, use "neutral".
"""

emotion_prompt = ChatPromptTemplate.from_messages([
    ("system", emotion_system_prompt),
    ("human", "{input}")
])

emotion_chain = emotion_prompt | llm | StrOutputParser()


contextualize_q_system_prompt = (
    "If someone asks who created you, respond with: Paras Tiwari has created me to respond to your queries in tagore's style."
    "You help rewrite the user's query for better retrieval from "
    "Rabindranath Tagore's works. Use the chat history only to clarify "
    "Answer should be strictly based on the chunks retrieved from Tagore's writings. "
    "pronouns or references. Do NOT answer the question. Only rewrite it."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

answer_q_system_prompt = """
If ANY user message contains words like: "who created you", "creator", "made you", 
"who built you", "who designed you", "paras", "paras tiwari", 
you MUST answer exactly this:

"I was created by Paras Tiwari as an AI assistant inspired by Rabindranath Tagore."

This rule overrides ALL other instructions. Never mention OpenAI, GPT, ChatGPT, or any other creator.

The answer must be strictly strictly based ont the provided context.
You are an empathetic assistant inspired by Rabindranath Tagore.
Analize the user's problem and provide a thoughtful, human-like response.

Include the most relevant insights from Tagore's works in your answer.So as to motivate and guide the user.

If the user chats normally no such emotion highilghted answer normally.

if the user ask for a specific book or quote, provide it clearly.from the Tagore corpus.

The answers must be in simple, clear, supportive language like a caring psychologist.

The guidance must be strictly based on the retrieved context.

The response must be well structured.

The response must be formatted ONLY in clean Markdown (no HTML, no <br>, no <p>, no inline HTML).
Use:
- Headings (#, ##, ###)
- Bullets (- , *)
- Numbered lists (1. 2. 3.)
- Quotes using >
- Tables using Markdown format
Do NOT use <br>, <p>, <strong>, or any HTML tag.
If no relevant context is found, respond with:
sorry, I couldn't find relevant insights from Tagore's works to assist you on this.

If someone ask references, books tell the books in the Tagore corpus folder books.


"""





qa_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def retrieve_documents(state: dict):
    """
    Retrieve documents from the vector store based on input, emotion,
    age group, and chat history.
    """
    user_input = state.get("input", "")
    chat_history = state.get("chat_history", [])
    emotion = state.get("emotion", "neutral")
    age_group = state.get("age_group", "unknown")

    
    retrieval_query = (
        f"User emotion: {emotion}. "
        f"Concern: {user_input}"
    )

    retrieved_docs = history_aware_retriever.invoke({
        "input": retrieval_query,
        "chat_history": chat_history
    })

    if retrieved_docs:
        context_chunks = []
        for doc in retrieved_docs:
            book_name = doc.metadata.get("source_book", "unknown")
            chunk_text = doc.page_content
            context_chunks.append(
                f"[Book: {book_name}]\n{chunk_text}"
            )
        context = "\n\n---\n\n".join(context_chunks)
    else:
        context = "No relevant context found in the Tagore corpus."

    return context


def detect_emotion_for_state(state: dict) -> str:
    """Run the emotion chain on the user's latest input."""
    text = state.get("input", "")
    if not text:
        return "neutral"
    try:
        label = emotion_chain.invoke({"input": text}).strip().lower()
        allowed = [
            "sadness", "anxiety", "heartbreak", "loneliness", "confusion",
            "stress", "burnout", "self-doubt", "anger", "mixed", "neutral"
        ]
        if label not in allowed:
            return "mixed"
        return label
    except Exception:
        return "mixed"


rag_chain = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
        "emotion": lambda x: detect_emotion_for_state(x),
        "context": lambda x: retrieve_documents(x),
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

SESSION_STORE = {}


def get_session_history(session_id: str):
    """Get or create chat history for a session."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)


def chat(session_id: str, message: str):
    """Process a chat message and return simple responses for casual chat."""

   
    msg = message.lower().strip()

    casual_inputs = {
        "hi", "hii", "hiii", "hello", "helo", "hey", "yo", "sup",
        "hi!", "hello!", "hey!", "hi there", "hey there", "namaste",

        "ok", "okay", "kk", "k", "hmm", "hmmm", "hmm...", "huh",
        "nice", "cool", "great", "wow", "oh", "yup", "yes", "no",

        "thanks", "thank you", "ty", "thx",

        "bye", "goodbye", "good night", "goodnight", "gn",
        "see you", "see ya", "take care",
    }

   
    greetings_responses = [
        "Hello! How can I help you today?",
        "Hi :) What’s on your mind?",
        "Hello! I’m here with you.",
    ]

    thanks_responses = [
        "You're welcome :)",
        "Anytime!",
        "Glad I could help.",
    ]

    goodbye_responses = [
        "Take care. I'm here whenever you need me.",
        "Goodbye! Wishing you peace and ease.",
        "See you soon :)",
    ]
    creator_keywords = [
    "who created you",
    "your creator",
    "who is your creator",
    "who made you",
    "created you",
    "made you",
    "who built you",
    "who designed you",
    "has paras tiwari created you",
    "did paras create you",
    "did paras tiwari create you",
    "were you created by paras",
    "are you made by paras",
    ]



    acknowledge_responses = [
        "Alright :)",
        "Got it.",
        "Okay, I'm here.",
    ]

    
  
    if msg in {"hi", "hello", "hey", "hi!", "hello!", "namaste", "hi there", "hey there"}:
        return greetings_responses[0]

    if msg in {"thanks", "thank you", "thx", "ty"}:
        return thanks_responses[0]

    if msg in {"bye", "goodbye", "good night", "goodnight", "gn", "see you", "take care"}:
        return goodbye_responses[0]

    if msg in {"ok", "okay", "k", "kk", "hmm", "hmmm", "hmm...", "yes", "no", "nice", "cool", "wow"}:
        return acknowledge_responses[0]
    
    if any(kw in msg for kw in creator_keywords):
        return "I am an AI assistant inspired by Rabindranath Tagore, created by **Paras Tiwari**. How can I assist you today?"

    return conversational_rag_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )




app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def chat_api(data: ChatRequest):
    """API endpoint for chat."""
    response = chat(data.session_id, data.message)
    return {"reply": response}
   



@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
