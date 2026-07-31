import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0;
        }

        .app-subtitle {
            font-size: 17px;
            color: #64748b;
            margin-bottom: 25px;
        }

        .video-card {
            background: white;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .chat-card {
            background: white;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .status-box {
            padding: 12px;
            border-radius: 10px;
            background: #ecfdf5;
            color: #166534;
            border: 1px solid #bbf7d0;
            margin-bottom: 15px;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


@st.cache_resource
def load_rag_chain():
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY was not found in the .env file."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.load_local(
        "youtube_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant for a YouTube video.

Answer the user's question only from the provided transcript context.

Do not use outside knowledge.

If the answer is not present in the transcript, reply:

I don't know based on the provided video.

Transcript context:
{context}

User question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
    )

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    return main_chain


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="app-title">🎬 YouTube RAG Chatbot</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="app-subtitle">
        Watch the video and ask questions based on its transcript.
    </p>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD RAG SYSTEM
# =========================================================

try:
    main_chain = load_rag_chain()
    rag_ready = True

except Exception as error:
    rag_ready = False
    st.error(f"Unable to load the RAG system: {error}")


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Settings")

    video_url = st.text_input(
        "YouTube video URL",
        value="https://www.youtube.com/watch?v=Gfr50f6ZBvo&t=11s",
        placeholder="Paste the YouTube video URL",
    )

    st.caption(
        "Use the same video whose transcript was used to create the FAISS index."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if rag_ready:
        st.success("RAG system connected")
    else:
        st.error("RAG system not connected")


# =========================================================
# MAIN LAYOUT
# =========================================================

video_column, chat_column = st.columns(
    [1.1, 1],
    gap="large",
)


# =========================================================
# VIDEO SECTION
# =========================================================

with video_column:
    st.subheader("Video")

    with st.container(border=True):
        if video_url and "youtube.com" in video_url or "youtu.be" in video_url:
            st.video(video_url)
        else:
            st.info("Paste a valid YouTube URL in the sidebar.")

        st.caption(
            "The chatbot answers from the transcript stored in your FAISS index."
        )


# =========================================================
# CHAT SECTION
# =========================================================

with chat_column:
    st.subheader("Chat with the video")

    chat_container = st.container(
        height=520,
        border=True,
    )

    with chat_container:
        if not st.session_state.messages:
            st.info(
                "Try asking: What is this video about?"
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the video..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not rag_ready:
                answer = (
                    "The RAG system is not connected. "
                    "Please check your API key and FAISS files."
                )

                st.error(answer)

            else:
                with st.spinner("Searching the transcript..."):
                    try:
                        answer = main_chain.invoke(question)
                        st.markdown(answer)

                    except Exception as error:
                        answer = f"Error: {error}"
                        st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )