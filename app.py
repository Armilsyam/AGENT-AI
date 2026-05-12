import os
import asyncio

# [TRICK PYTHON 3.14] Memaksa sistem menggunakan Protobuf versi murni (Pure Python) 
# Ini sangat penting agar terhindar dari error 'Metaclasses with custom tp_new' di Python 3.14
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
# TAMBAHAN BARU: Import Streamlit Callback agar proses berpikir terlihat live & satset
from langchain_community.callbacks import StreamlitCallbackHandler

# ==========================================
# 1. PENGATURAN HALAMAN
# ==========================================
st.set_page_config(page_title="AI Agent Super - Asisten Pribadi", page_icon="⚡", layout="wide")
st.title("⚡ AI Agent Super Cerdas (Versi Ngebut)")
st.markdown("Berjalan dengan mesin AI modern, memori pintar, dan fitur Live Tracking.")

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
    # [TRICK ASYNCIO] Mengatasi error 'no current event loop' di Streamlit
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Otak Utama (Gemini 1.5 Flash - Menggunakan tag latest agar tidak Error 404)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.3,
        streaming=True  # TAMBAHAN: Biar hasil ketikan keluar lebih cepat
    )
    
    # Alat 1: Pencarian Internet Gratis
    search_tool = DuckDuckGoSearchRun(
        name="Search_Internet",
        description="Gunakan ini HANYA JIKA butuh fakta/berita terbaru hari ini. Jika tidak tahu, cari di sini."
    )
    tools = [search_tool]
    
    # Prompt Modern
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah AI Agent Super cerdas. Tugasmu adalah membantu dengan CEPAT, AKURAT, dan SATSET. Jangan terlalu banyak berbasa-basi. Jika butuh data internet, gunakan tool Search_Internet. Jika tool gagal/lama, langsung jawab sebisamu dengan pengetahuanmu."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Merakit Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # TAMBAHAN BARU: Membatasi waktu agar AI tidak nyangkut memikirkan hal yang sama berulang kali
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3,      # Maksimal 3 kali putaran mikir
        max_execution_time=15  # Waktu maksimal pencarian tool adalah 15 detik
    )
    
    return agent_executor

agent_executor = setup_agent()

# ==========================================
# 4. MANAJEMEN PERCAKAPAN & MEMORI
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo Bos! Mesin jet sudah dinyalakan. Siap mengerjakan tugas dengan satset!"}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. INPUT PENGGUNA & PROSES AI
# ==========================================
user_query = st.chat_input("Ketik tugas Anda di sini...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        # TAMBAHAN BARU: Memunculkan jejak pemikiran AI secara Live (bukan cuma loading muter)
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
        try:
            # Eksekusi agen dengan callback untuk efek live dan transparan
            response = agent_executor.invoke(
                {
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                },
                {"callbacks": [st_callback]}
            )
            
            output_text = response["output"]
            st.markdown(output_text)
            
            # Simpan ke history Streamlit
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            
            # Simpan ke memori LangChain (Maksimal 10 percakapan)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=output_text))
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
                
        except Exception as e:
            error_msg = f"⚠️ Proses dihentikan karena server internet terlalu lama merespon atau error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
