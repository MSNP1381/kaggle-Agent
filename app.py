import copy
from pprint import pprint
import black
import streamlit as st
import time
import os
import httpx
from dotenv import load_dotenv
from agent import KaggleProblemSolver
from langfuse.callback import CallbackHandler
from langgraph.graph.graph import CompiledGraph

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False

def main1():
    if st.session_state.running:
        return
    
    st.session_state.running = True

    # Streamlit interface
    st.session_state.chats.write("is running")
    ev = st.session_state.ev
    event = next(ev)
    
    # Debug print
    # pprint(event)
    
    p = event
    k = list(p)[0]
    st.session_state.chats.subheader(f"Node : {k}")
    v = p[k]
    # pprint(v)
    # print(p.keys())
    if "enhanced_task" in v:
        v['enhanced_task']=v['enhanced_task'].dict()
        # print(v['enhanced_task'])
    if 'task_codes_results' in v :
        l=[]
        for i in v['task_codes_results']:
            l.append((i[0].dict(),i[1].dict(),i[2]))
        v['task_codes_results']=l   
        print(l) 
    
    st.session_state.chats.json(v)
    # st.session_state.chats.subheader(f"Node: {k}")
    
    st.session_state.running = False
    if st.session_state.go:
        main1()

def start():
    if 'ev' not in st.session_state:
        st.session_state.running = True
        st.session_state.go=False

        st.title("LangGraph Kaggle Agent")
        load_dotenv()
        
        proxy = httpx.Client(proxies=os.getenv("HTTP_PROXY_URL"))
        
        langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
            session_id=f"session-{int(time.time())}",
        )
        
        config = {
            "configurable": {"thread_id": str(int(time.time()))},
            "callbacks": [langfuse_handler],
            "recursion_limit": 50,
        }
        
        solver = KaggleProblemSolver(config, proxy)
        graph = solver.compile()
        s = solver._init_state()
        ev = graph.stream(s, config=config, stream_mode="updates")
        
        st.session_state.graph = graph
        st.session_state.ev = ev
        st.session_state.cfg = config
        st.session_state.chats = st.empty()
        
        st.session_state.running = False
        st.session_state.chats.subheader("Ready to run")

# Create a container for the input and button
input_container = st.container()

if input_container.button("pause/go", use_container_width=True, disabled=st.session_state.running):
    st.session_state.go=not st.session_state.go
    main1()
if input_container.button("Next", use_container_width=True, on_click=main1, disabled=st.session_state.running):
    st.write("Button clicked")

chat_container = st.container()
st.session_state.chats = chat_container

start()
