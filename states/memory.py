import os
from typing import Dict, List, Optional, Tuple, Union

from bson import ObjectId
from injector import inject
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import state2retrieve_doc

RETRIVE_PROMPT = PromptTemplate.from_template(
    """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
you get a task and generate data relevant to task and insights, think before answering and show your thinking process step by step
Question: {question}

Context: {context}
"""
)

UPDATE_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
You are an AI assistant tasked with analyzing and summarizing the results of code executions in a machine learning project. Your goal is to provide insights and maintain an up-to-date summary of the project's progress.

Current Summary: {current_summary}

New Information:
Task: {task}
Code: {code}
Result: {result}

Please update the summary based on this new information. Consider the following:
1. What was the purpose of this task?
2. Was the code execution successful? If not, what were the issues?
3. What insights can be drawn from the results?
4. How does this task contribute to the overall project goals?
5. Are there any recommendations for future tasks or improvements?

Provide a concise yet comprehensive update to the summary, incorporating the most important points from this new information. The updated summary should give a clear picture of the project's current state and progress.

Updated Summary:
"""
)


class WeightedMemory:
    def __init__(self, max_items=5):
        self.memories: List[Tuple[str, float]] = []
        self.max_items = max_items

    def add_memory(self, text: str, importance: float = 1.0):
        self.memories.append((text, importance))
        self.memories.sort(key=lambda x: x[1], reverse=True)
        if len(self.memories) > self.max_items:
            self.memories = self.memories[: self.max_items]

    def get_memories(self) -> str:
        return "\n".join([memory[0] for memory in self.memories])


class MemoryAgent:
    @inject
    def __init__(self, llm, mongo):
        self.task_results_dict = {}
        self.docs_retriever = None
        self.docs_vectorstore = None
        self.base_url = os.getenv("BASE_URL", "https://api.avalapis.ir/v1")
        self.embeddings = OpenAIEmbeddings(base_url=self.base_url)
        self.llm = llm

        self.client = mongo
        self.db = self.client["kaggle_agent"]
        self.collection = self.db["long_term_memory"]
        self.examples_collection = self.db["examples"]

        self.results_summary = ""

        # Replace MongoDBAtlasVectorSearch with Chroma
        self.long_term_memory = Chroma(
            collection_name="long_term_memory", embedding_function=self.embeddings
        )

        self.short_term_memory = WeightedMemory()
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        return ConversationalRetrievalChain.from_llm(
            self.llm, self.long_term_memory.as_retriever(), return_source_documents=True
        )

    def update_summary(self, task: str, code: str, result: str):
        prompt = UPDATE_SUMMARY_PROMPT.format(
            current_summary=self.results_summary, task=task, code=code, result=result
        )

        updated_summary = self.llm.invoke(prompt)
        self.results_summary = updated_summary

        # Add the update to short-term memory as well
        self.add_to_short_term_memory(
            f"Task: {task}\nResult Summary: {updated_summary}", importance=2.0
        )

        return updated_summary

    def add_example(self, task: str, code: str, result: str) -> str:
        """
        Add a new example (task, code, result) to both MongoDB and Chroma.
        """
        self.update_summary(task, code, result)
        # Add to MongoDB
        doc = {
            "task": task,
            "code": code,
            "result": result,
        }
        doc["_id"] = str(ObjectId())

        # Add to Chroma
        self._ensure_examples_in_chroma()
        self.collection.insert_one(doc)
        self.examples_vectorstore.add_documents(
            documents=[
                Document(
                    page_content=task,
                    metadata={"source": "enhanced task", "id": doc["_id"]},
                )
            ],
        )

    def get_few_shots(self, task: str, n: int = 4) -> List[Dict[str, str]]:
        """
        Retrieve n most similar (task, code, result) examples based on the given task using Chroma.
        """
        try:
            # Ensure the examples are in Chroma
            self._ensure_examples_in_chroma()

            # Get the embedding for the current task
            task_embedding = self.embeddings.embed_query(task)

            # Perform a vector similarity search using Chroma
            similar_examples = self.examples_vectorstore.similarity_search_by_vector(
                task_embedding, k=n
            )

            # Format the results
            return [
                self.collection.find_one({"_id": doc.metadata["id"]})
                for doc in similar_examples
            ]
        except Exception:
            return []

    def _ensure_examples_in_chroma(self):
        """
        Ensure all examples are stored in Chroma.
        """
        if not hasattr(self, "examples_vectorstore"):
            # Initialize Chroma for examples if it doesn't exist
            self.examples_vectorstore = Chroma(
                collection_name="examples", embedding_function=self.embeddings
            )

            # Add all examples from MongoDB to Chroma
            for example in self.examples_collection.find():
                self.examples_vectorstore.add_texts(
                    texts=[example["task"]],
                    metadatas=[{"source": "examples"}],
                    ids=[str(example["_id"])],
                )

    def init_doc_retrieve(self):
        documents = state2retrieve_doc()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        self.docs_vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embeddings
        )
        self.docs_vectorstore.as_retriever()
        self.docs_retriever = self.docs_vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.docs_rag_chain = (
            {
                "context": self.docs_retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RETRIVE_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask_docs(self, question):
        self.docs_rag_chain.invoke(question)

    def ask(self, question: str, doc_type: Optional[str] = None) -> str:
        context = self.short_term_memory.get_memories()
        full_question = f"Context: {context}\n\nQuestion: {question}"

        if doc_type:
            retriever = self.long_term_memory.as_retriever(
                search_kwargs={"filter": {"doc_type": doc_type}}
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm, retriever, return_source_documents=True
            )
            result = qa_chain({"question": full_question})
        else:
            result = self.qa_chain({"question": full_question})

        self.short_term_memory.add_memory(f"Q: {question}\nA: {result['answer']}")

        return result["answer"]

    def add_document(
        self, content: str, doc_type: str, metadata: Optional[Dict] = None
    ) -> str:  # Changed return type to str
        if metadata is None:
            metadata = {}

        metadata["doc_type"] = doc_type

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(content)

        # Add texts to Chroma
        ids = self.long_term_memory.add_texts(
            texts, metadatas=[{"doc_type": doc_type, **metadata} for _ in texts]
        )

        return ids[0]  # Return the first ID as a string

    def load_document(
        self, doc_id: str
    ) -> Union[Dict, None]:  # Changed parameter type to str
        # Retrieve the document from Chroma
        results = self.long_term_memory.get([doc_id])
        if results and results["documents"]:
            return {
                "content": results["documents"][0],
                "doc_type": results["metadatas"][0]["doc_type"],
                "metadata": results["metadatas"][0],
            }
        return None

    def add_to_short_term_memory(self, text: str, importance: float = 1.0):
        self.short_term_memory.add_memory(text, importance)

    def search_documents(
        self, query: str, doc_type: Optional[str] = None, k: int = 5
    ) -> List[Document]:
        if doc_type:
            return self.long_term_memory.similarity_search(
                query, k=k, filter={"doc_type": doc_type}
            )
        else:
            return self.long_term_memory.similarity_search(query, k=k)

    def list_documents(self, doc_type: Optional[str] = None) -> List[Dict]:
        # This method needs to be implemented differently for Chroma
        # As Chroma doesn't have a direct method to list all documents, we'll need to use a workaround
        if doc_type:
            results = self.long_term_memory.similarity_search(
                "", filter={"doc_type": doc_type}, k=1000
            )
        else:
            results = self.long_term_memory.similarity_search("", k=1000)

        return [
            {
                "id": doc.metadata.get("id", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "metadata": doc.metadata,
            }
            for doc in results
        ]


# Usage example
if __name__ == "__main__":
    agent = MemoryAgent(
        api_key="your-openai-api-key",
        mongo_uri="your-mongodb-uri",
        db_name="your_db_name",
        collection_name="your_collection_name",
    )

    # Add documents to MongoDB
    doc1_id = agent.add_document(
        "AI and machine learning are transforming various industries...",
        "tech_report",
        {"author": "John Doe"},
    )
    doc2_id = agent.add_document(
        "Q3 financial results show a 15% increase in revenue...",
        "financial_report",
        {"quarter": "Q3", "year": "2023"},
    )

    # List all documents
    all_docs = agent.list_documents()
    print(f"Total documents: {len(all_docs)}")

    # Load a specific document
    loaded_doc = agent.load_document(doc1_id)
    if loaded_doc:
        print(f"Loaded document type: {loaded_doc['doc_type']}")

    # Ask questions about specific document types
    print(agent.ask("What are the main tech trends?", doc_type="tech_report"))
    print(agent.ask("What are the financial projections?", doc_type="financial_report"))

    # Ask a general question across all documents
    print(agent.ask("What are the key points from all documents?"))

    # Add important information to short-term memory

    # Search for specific information
    tech_docs = agent.search_documents("AI advancements", doc_type="tech_report")
    all_docs = agent.search_documents("company strategy")

    print(
        f"Found {len(tech_docs)} tech documents and {len(all_docs)} total relevant documents."
    )
