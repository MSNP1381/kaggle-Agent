import streamlit as st
import time
import os
import httpx
from dotenv import load_dotenv
from agent import KaggleProblemSolver
from langfuse.callback import CallbackHandler
from logging_module import log_it


# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.use_log = False


def transform(event, use_log):
    if use_log:
        p = event
        k = list(p)[0]
        data_reuslt = log_it.log_it(p[k], k)
        # pprint(data_reuslt)
        return data_reuslt

    else:
        p = event

        k = list(p)[0]

        v = p[k]
        if "enhanced_tasks" in v:
            v["enhanced_tasks"] = list(map(lambda x: x.dict(), v["enhanced_tasks"]))

        if "task_codes_results" in v:
            task_codes_results_list = []
            for i in v["task_codes_results"]:
                task_codes_results_list.append((i[0].dict(), i[1].dict(), i[2]))
            v["task_codes_results"] = task_codes_results_list
        return v


def main1():
    if st.session_state.running:
        return

    st.session_state.running = True
    # Streamlit interface
    # st.session_state.chats.write("is running")
    ev = st.session_state.ev
    event = next(ev)

    # Debug print
    # pprint(event)

    # v = transform(event)
    k = list(event)[0]
    # used_log=transform(event,True)
    classic = transform(event, False)
    st.session_state.messages.append({"role": "ai", "title": k, "message": [classic]})
    # st.session_state.chats.json(v)
    # st.session_state.chats.subheader(f"Node: {k}")
    i = {"role": "ai", "title": k, "message": [classic]}
    with st.chat_message(i["role"]):
        st.subheader(f"Node : {i['title']}")
        if i["message"]:
            if st.session_state.use_log:
                st.json(i["message"][0])
            else:
                st.json(i["message"][0])
    st.session_state.running = False
    if st.session_state.go:
        main1()


def start(use_langfuse=False, use_pg_persistence=True):
    if "ev" not in st.session_state:
        st.session_state.running = True
        st.session_state.go = False
        st.session_state.use_log = True

        st.title("LangGraph Kaggle Agent")
        load_dotenv(override=True)

        proxy = httpx.Client(proxies=os.getenv("HTTP_PROXY_URL"))
        config = {
            "configurable": {"thread_id": str(int(time.time()))},
            "recursion_limit": 50,
        }

        # checkpointer = PostgresSaver(sync_connection=pool)
        # checkpointer.create_tables(pool)
        if use_langfuse:
            langfuse_handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
                session_id=f"session-{int(time.time())}",
            )
            config["callbacks"] = [langfuse_handler]

        solver = KaggleProblemSolver(config, proxy)
        graph = solver.compile(None)
        s = solver._init_state()

        ev = graph.stream(s, config=config, stream_mode="updates")
        # for i in ev:
        # pass
        # with open("w.txt",'a') as f:
        # pprint(i)
        # print(pformat(i),file= f)
        print("done")
        st.session_state.graph = graph
        st.session_state.ev = ev
        st.session_state.cfg = config
        # st.session_state.chats = st.empty()

        st.session_state.running = False
        st.session_state.messages.append(
            {"title": "Ready to run", "role": "ai", "message": None}
        )


# Create a container for the input and button

if "messages" not in st.session_state:
    st.session_state.messages = []

side_bar = st.sidebar

if side_bar.button(
    "pause/go", use_container_width=True, disabled=st.session_state.running
):
    st.session_state.go = not st.session_state.go
    main1()
if side_bar.button(
    "Next", use_container_width=True, on_click=main1, disabled=st.session_state.running
):
    st.write("Button clicked")


def toggle_it():
    if st.session_state.use_log:
        st.session_state.use_log = False
    else:
        st.session_state.use_log = True


side_bar.toggle("use log like", st.session_state.use_log, on_change=toggle_it)

print(st.session_state.use_log)

chats = st.container(border=True)

start()
