# product_search_service.py
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1) Recreate the exact same embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2) Load the persisted Chroma index (no re-embedding!)
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# 3) Build your retriever + QA chain
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-ul2",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.7,
    max_new_tokens=256,
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a product-search assistant. Use the information from the retrieved products below
to answer the user's question as precisely as possible.

Product Info:
{context}

Question: {question}

Answer:
""".strip()
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
# 3) THIS is the missing piece: expose a single-call function
import traceback, time

def answer_query(query: str) -> dict:
    print("answer_query called with:", query)
    start = time.time()
    try:
        out = qa_chain.invoke({"query": query})
        print(f"⏱️  HF round-trip: {time.time() - start:.1f}s")
        return out
    except Exception as e:
        print("‼️  invoke raised:")
        traceback.print_exc()
        return {"result": "⚠️ error – see server log", "source_documents": []}

