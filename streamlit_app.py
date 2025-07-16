# streamlit_app.py
import streamlit as st
import pandas as pd
from product_search_service import answer_query
from dotenv import load_dotenv
load_dotenv()

print("Streamlit app started")  # Debug

st.set_page_config(page_title="🔍 Product Search", layout="wide")
st.title("🔍 Product Search Agent")

# 1) Always show the input box
query = st.text_input(
    "Ask me about products…",
    placeholder="e.g. Nike shoes under ₹2000"
)
print("Query input rendered")  # Debug

# 2) Run the chain when the user enters something
if query:
    print(f"User entered query: {query}")  # Debug
    with st.spinner("Searching…"):
        try:
            output = answer_query(query)
            print("answer_query executed successfully")  # Debug
        except Exception as e:
            print(f"Error in answer_query: {e}")  # Debug
            st.error(f"Error: {e}")
            output = None

    # 3) Display the LLM's answer
    if output:
        st.subheader("Answer")
        st.write(output["result"])
        print("Displayed answer")  # Debug

        # 4) Build & display a DataFrame of source docs
        rows = []
        for doc in output["source_documents"]:
            m = doc.metadata
            snippet = doc.page_content[:150].rstrip() + ("..." if len(doc.page_content)>150 else "")
            rows.append({
                "Name":      m["name"],
                "Brand":     m["brand"],
                "Price (₹)": m["price"],
                "Category":  m["category"],
                "Available": m["available"],
                "URL":       m["url"],
                "Snippet":   snippet
            })
        df = pd.DataFrame(rows)
        st.subheader("Matched Products")
        st.dataframe(df, use_container_width=True)
        print("Displayed results table")  # Debug
