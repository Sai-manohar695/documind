import streamlit as st
import requests
import os

# Backend URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 DocuMind")
st.subheader("Intelligent Document Q&A Engine")
st.divider()

# ----------- Sidebar -----------
with st.sidebar:
    st.header("📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing PDF... This may take a moment."):
                # Send file to backend
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Document processed!")
                    st.info(f"📊 Created {data['chunks_created']} chunks")
                    
                    # Store doc_id in session
                    st.session_state["doc_id"] = data["doc_id"]
                    st.session_state["filename"] = data["filename"]
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
    
    st.divider()
    
    # Show current document
    if "filename" in st.session_state:
        st.success(f"📄 Active: {st.session_state['filename']}")
    
    # Stats
    if st.button("📊 Show Stats"):
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            st.metric("Total Chunks", stats["total_chunks"])
    
    st.divider()
    
    # Delete document
    if "doc_id" in st.session_state:
        if st.button("🗑️ Delete Document", type="secondary"):
            response = requests.delete(
                f"{API_URL}/document",
                json={"doc_id": st.session_state["doc_id"]}
            )
            if response.status_code == 200:
                st.success("Document deleted!")
                del st.session_state["doc_id"]
                del st.session_state["filename"]
            else:
                st.error("Failed to delete document.")


# ----------- Main Area -----------

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show confidence and sources for assistant messages
        if message["role"] == "assistant" and "confidence" in message:
            conf = message["confidence"]
            
            # Confidence badge
            if conf["level"] == "high":
                st.success(f"🟢 Confidence: {conf['level'].upper()} ({conf['score']})")
            elif conf["level"] == "medium":
                st.warning(f"🟡 Confidence: {conf['level'].upper()} ({conf['score']})")
            else:
                st.error(f"🔴 Confidence: {conf['level'].upper()} ({conf['score']})")
            
            st.caption(conf["message"])
            
            # Show sources
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (Score: {source['score']})")
                        st.text(source["text"])
                        st.divider()


# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    
    # Check if document is uploaded
    if "doc_id" not in st.session_state:
        st.warning("⚠️ Please upload a PDF document first using the sidebar.")
    else:
        # Add user message to history
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get answer from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "query": prompt,
                        "doc_id": st.session_state["doc_id"]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    confidence = data["confidence"]
                    sources = data["sources"]
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display confidence
                    if confidence["level"] == "high":
                        st.success(f"🟢 Confidence: {confidence['level'].upper()} ({confidence['score']})")
                    elif confidence["level"] == "medium":
                        st.warning(f"🟡 Confidence: {confidence['level'].upper()} ({confidence['score']})")
                    else:
                        st.error(f"🔴 Confidence: {confidence['level'].upper()} ({confidence['score']})")
                    
                    st.caption(confidence["message"])
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}** (Score: {source['score']})")
                                st.text(source["text"])
                                st.divider()
                    
                    # Save to history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "confidence": confidence,
                        "sources": sources
                    })
                    
                else:
                    st.error("❌ Failed to get an answer. Please try again.")