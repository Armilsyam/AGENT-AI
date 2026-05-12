import os

# [TRICK PYTHON 3.14] Memaksa sistem menggunakan Protobuf versi murni (Pure Python) 
# Ini sangat penting agar terhindar dari error 'Metaclasses with custom tp_new' di Python 3.14
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# ==========================================
# 1. PENGATURAN HALAMAN
# ==========================================
st.set_page_config(page_title="AI Agent Super - Asisten Pribadi", page_icon="🤖", layout="wide")
st.title("⚡ AI Agent Super Cerdas (Python 3.14 Ready)")
st.markdown("Berjalan mulus di Python 3.14 dengan mesin AI modern dan kemampuan pencarian internet.")

# ==========================================
# 2. MENGAMBIL KUNCI RAHASIA (API KEY)
# ==========================================
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.warning("⚠️ GOOGLE_API_KEY belum diatur di Streamlit Secrets! Silakan atur terlebih dahulu.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ==========================================
# 3. INISIALISASI OTAK AI & ALAT (TOOLS) MODERN
# ==========================================
@st.cache_resource
def setup_agent():
    # Otak Utama (Gemini 1.5 Flash - Cepat & Pintar)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.3
    )
    
    # Alat 1: Pencarian Internet Gratis
    search_tool = DuckDuckGoSearchRun(
        name="Search_Internet",
        description="Gunakan ini untuk mencari informasi, berita, atau fakta terbaru di internet."
    )
    tools = [search_tool]
    
    # Prompt Modern (Instruksi Kepribadian)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah AI Agent Super cerdas milik pengguna. Tugasmu adalah membantu mengerjakan APAPUN. Jika ditanya informasi terkini, GUNAKAN tool Search_Internet. Berbicaralah dalam bahasa Indonesia yang asik."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Merakit Agent Modern (Pengganti versi lama yang error)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

agent_executor = setup_agent()

# ==========================================
# 4. MANAJEMEN PERCAKAPAN & MEMORI
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo Bos! Sistem sudah di-upgrade penuh untuk Python 3.14. Apa tugas kita hari ini?"}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. INPUT PENGGUNA & PROSES AI
# ==========================================
user_query = st.chat_input("Perintahkan tugas Anda di sini...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Sedang berpikir dan mencari di internet jika perlu... ⏳"):
            try:
                # Eksekusi agen
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                output_text = response["output"]
                st.markdown(output_text)
                
                # Simpan ke history Streamlit
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
                # Simpan ke memori LangChain (Maksimal 10 percakapan agar tidak berat)
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=output_text))
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                    
            except Exception as e:
                error_msg = f"Mohon maaf, terjadi kesalahan sistem: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
