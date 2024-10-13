from typing import List, Dict, Any
from collections import deque
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import faiss
from injector import inject
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class MemoryAgent:
    @inject
    def __init__(
        self,
        llm: ChatOpenAI,
        max_short_term_size: int = 10,
        long_term_file: str = "long_term_memory.json",
    ):
        self.short_term_memory: deque = deque(maxlen=max_short_term_size)
        self.long_term_memory: List[Dict[str, Any]] = []
        self.long_term_file = long_term_file

        # Initialize LLM and embeddings
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(base_url="https://api.avalai.ir/v1")

        # Initialize FAISS vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Initialize VectorStoreRetrieverMemory
        self.retriever_memory = VectorStoreRetrieverMemory(
            retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        )

        # Load long-term memory after initializing vector store
        self.load_long_term_memory()

    def add_to_short_term_memory(self, content: str):
        self.short_term_memory.append(content)

    def add_to_long_term_memory(self, content: Dict[str, Any]):
        
        self.long_term_memory.append(content)
        self.save_long_term_memory()
        self._add_to_vector_store(content)

    def _add_to_vector_store(self, content: Dict[str, Any]):
        text = json.dumps(content)
        
        self.vector_store.add_documents([Document(page_content=text)])

    def get_relevant_context(self, query: str, max_results: int = 5) -> List[str]:
        # Use VectorStoreRetrieverMemory to get relevant context
        relevant_docs = self.retriever_memory.load_memory_variables({"prompt": query})
        
        # Extract and rank the most relevant pieces of information
        relevant_info = []
        for doc in relevant_docs.get("memory", []):
            relevant_info.append(self._extract_relevant_info(query, doc))
        
        # Sort by relevance score and return the top results
        relevant_info.sort(key=lambda x: x['score'], reverse=True)
        return [item['content'] for item in relevant_info[:max_results]]

    def _extract_relevant_info(self, query: str, content: str) -> Dict[str, Any]:
        prompt = f"""
        Given the query: "{query}"
        And the following content:
        {content}

        Extract the most relevant information from the content that answers the query.
        Also, provide a relevance score between 0 and 1, where 1 is highly relevant and 0 is not relevant at all.

        Output format:
        Relevant information: <extracted_info>
        Relevance score: <score>
        """

        response = self.llm.predict(prompt)
        
        # Parse the response
        info = response.split("Relevant information:")[1].split("Relevance score:")[0].strip()
        score = float(response.split("Relevance score:")[1].strip())

        return {"content": info, "score": score}

    def load_long_term_memory(self):
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, 'r') as f:
                self.long_term_memory = json.load(f)
            # Add existing memory to vector store
            for item in self.long_term_memory:
                self._add_to_vector_store(item)

    def save_long_term_memory(self):
        with open(self.long_term_file, 'w') as f:
            json.dump(self.long_term_memory, f)

    def add_to_conversation(self, role: str, content: str):
        self.add_to_short_term_memory(f"{role}: {content}")

    def get_conversation_history(self, max_messages: int = 10) -> List[str]:
        return list(self.short_term_memory)[-max_messages:]

    def get_format_instructions(self) -> str:
        return """
        Please provide your response in the following format:
        
        Task: [Original task description]
        
        Thought: [Your reasoning about the task]
        
        Actions:
        1. [First action]
        2. [Second action]
        ...
        
        Observation: [Result or outcome of the actions]
        
        Final Thought: [Concluding reasoning]
        
        Enhanced Task: [The refined task description]
        """

    def clear_short_term_memory(self):
        self.short_term_memory.clear()

    def clear_long_term_memory(self):
        self.long_term_memory.clear()
        self.save_long_term_memory()
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.retriever_memory = VectorStoreRetrieverMemory(
            retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        )

    def get_memory_summary(self) -> Dict[str, Any]:
        return {
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "long_term_file": self.long_term_file
        }

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory_agent = MemoryAgent(llm=llm)
    print(memory_agent.get_memory_summary())
