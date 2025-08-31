# 📄 Chat-With-Your-PDF

An **AI-powered PDF Reader Chatbot** that allows users to upload PDFs and interact with their content in real-time.  
The system leverages **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and **LangChain** for document-grounded responses, with an interactive **Streamlit UI** for seamless user experience.

---

## 🚀 Features

- 📚 **RAG Pipeline**:  
  Implemented with `RecursiveCharacterTextSplitter` for text chunking and efficient context retrieval.
  
- 🔍 **FAISS Vector Store**:  
  High-performance vector database for fast and scalable similarity search.

- 🏷️ **Metadata Tagging**:  
  Preserves document structure and contextual information for better grounding.

- 🤖 **LangChain Integration**:  
  Powers prompt engineering and retrieval-based augmentation for improved LLM responses.

- 🧠 **Conversational Memory**:  
  Remembers previous queries and responses, enabling smooth multi-turn conversations.

- 🌐 **Streamlit Web App**:  
  Upload PDFs, query documents, and reset chat history in real time with a user-friendly UI.

---

## 📂 Project Structure

```bash
Chat-With-Your-Pdf/
├── .src/                       # Source code files
├── .venv/                      # Virtual environment
├── document/                   # Uploaded PDFs
├── faiss_index/                # Stored FAISS vector index
├── README.md                   # Project documentation
├── chat_history.json           # Stores conversational memory
└── requirements.txt            # Python dependencies
