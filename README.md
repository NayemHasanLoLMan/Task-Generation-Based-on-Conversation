## 🧠 Task Generation Based on Conversation

An intelligent task generation system powered by **OpenAI**, **Pinecone**, and the **Raag knowledge base**. This project allows users to have meaningful conversations with uploaded documents (e.g., books), and automatically generates well-structured tasks including **core objectives**, **subtasks**, **timelines**, and **descriptions** — all derived contextually from the discussion.

---

### 📌 Project Description

This project transforms the way we interact with knowledge. Using a combination of **OpenAI’s LLM**, **Pinecone's vector database**, and **Raag’s knowledge management system**, the bot understands long documents and generates actionable task plans based on real-time conversations.

You can upload a book or document, talk to the system about your goals, and it will intelligently break down the content into relevant tasks, milestones, and timelines. Ideal for students, researchers, project managers, or lifelong learners.

---

### ⚙️ How It Works

1. **Document Ingestion**  
   - Upload a book or knowledge source.  
   - Content is chunked and embedded into **Pinecone** for fast semantic search.

2. **Conversational Interface**  
   - Chat with the system about your goals, plans, or interests.  
   - The bot uses **OpenAI GPT** to understand the context and retrieve relevant content.

3. **Task Generation Engine**  
   - Extracts core objectives based on the conversation.  
   - Breaks down into subtasks with estimated timelines and detailed descriptions.

---

### ✨ Key Features

- ✅ Chat-based interface for interaction with books/documents  
- ✅ Vector search with **Pinecone** for fast content retrieval  
- ✅ Real-time task plan generation using **OpenAI GPT**  
- ✅ Timeline and milestone suggestions  
- ✅ Modular architecture for extensibility  
- ✅ Supports multi-level task trees (core → subtasks)

---

### 🛠 Tech Stack

- **Language**: Python  
- **AI Model**: OpenAI GPT-4 (via API)  
- **Vector Database**: Pinecone  
- **Knowledge Base Integration**: Raag system  
- **Embeddings**: `text-embedding-ada-002`  
- **Interface**: CLI / (Optional: Streamlit / Web-based UI)

---

### 🚀 Getting Started

#### Prerequisites

- Python 3.8+
- OpenAI API Key
- Pinecone API Key
- Raag knowledge base access (optional)

#### Installation


    git clone https://github.com/your-username/Task-Generation-Based-on-Conversation.git
    cd Task-Generation-Based-on-Conversation
    pip install -r requirements.txt


#### Setup .env File

    OPENAI_API_KEY=your-openai-key
    PINECONE_API_KEY=your-pinecone-key
    PINECONE_ENVIRONMENT=your-pinecone-environment

###🧠 Future Enhancements

- UI dashboard for task tracking
- Integration with calendar/task apps (Notion, Trello, Google Calendar)
- Multi-document knowledge fusion
- Personalized learning/task strategies


###📄 License

This project is licensed under the MIT License.
