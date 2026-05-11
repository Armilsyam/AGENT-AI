import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

# ==========================================
# 1. PENGATURAN HALAMAN & TAMPILAN
# ==========================================
st.set_page_config(page_title="AI Agent Super - Asisten Pribadi", page_icon="🤖", layout="wide")
st.title("⚡ AI Agent Super Cerdas & Cloud-Based")
st.markdown("Asisten ini tidak membebani komputer lokal Anda. Ia memiliki kemampuan mencari di internet dan mengingat percakapan.")

# ==========================================
# 2. MENGAMBIL KUNCI RAHASIA (API KEY)
# ==========================================
# Mengambil API Key dari Streamlit Secrets (Aman di Cloud)
api_key = st.secrets.get("AIzaSyBSudsAHWsMqvms3aWLElL1n4CPppDfxJI")

if not api_key:
    st.warning("⚠️ GOOGLE_API_KEY belum diatur di Streamlit Secrets! Silakan atur terlebih dahulu sesuai panduan.")
    st.stop()

os.environ["AIzaSyBSudsAHWsMqvms3aWLElL1n4CPppDfxJI"] = api_key

# ==========================================
# 3. INISIALISASI OTAK AI & ALAT (TOOLS)
# ==========================================
# Menggunakan Gemini 1.5 Flash (Sangat cepat, pintar, dan gratis limit besar)
@st.cache_resource
def setup_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.3, # 0.3 agar cerdas tapi tetap logis/faktual
        convert_system_message_to_human=True
    )
    
    # Alat 1: Pencarian Internet Gratis
    search_tool = DuckDuckGoSearchRun(name="Search_Internet")
    tools = [search_tool]
    
    # Memori: Mengingat 5 percakapan terakhir agar tidak berat
    memory = ConversationBufferWindowMemory(
        memory_key="memory", 
        return_messages=True, 
        k=5
    )
    
    # Instruksi Sistem (Kepribadian AI)
    system_message = SystemMessage(
        content="""Kamu adalah AI Agent Super cerdas milik pengguna. Tugasmu adalah membantu mengerjakan APAPUN tugas yang diberikan dengan efisien, cepat, dan akurat. 
        Jika kamu ditanya tentang informasi terkini atau fakta yang tidak kamu ketahui, GUNAKAN tool Search_Internet. 
        Berbicaralah dalam bahasa Indonesia yang profesional namun asik."""
    )
    
    # Merakit Agent
    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={
            "system_message": system_message,
        },
        verbose=True, # Untuk melihat proses berpikir di console
        handle_parsing_errors=True
    )
    return agent_chain

agent = setup_agent()

# ==========================================
# 4. MANAJEMEN PERCAKAPAN (CHAT HISTORY)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo Bos! Saya AI Agent Super Anda yang berjalan 100% di Cloud. Apa yang bisa saya selesaikan untuk Anda hari ini?"}
    ]

# Tampilkan history chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. INPUT PENGGUNA & PROSES AI
# ==========================================
user_query = st.chat_input("Perintahkan tugas Anda di sini...")

if user_query:
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Proses oleh AI Agent
    with st.chat_message("assistant"):
        with st.spinner("Sedang berpikir dan mengeksekusi tugas secara cloud... ⏳"):
            try:
                # Menjalankan agen
                response = agent.run(user_query)
                st.markdown(response)
                # Simpan ke history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Mohon maaf, terjadi kesalahan pada jaringan/server: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
