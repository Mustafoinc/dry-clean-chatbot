import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Dry Cleaning Policy Assistant",
    page_icon="🧼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS - MATCHING PERSIPICO.COM STYLE
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #1a1a1a;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #2c5aa0;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
        border-left: 4px solid #2c5aa0;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
        border-left: 4px solid #666;
    }
    
    /* Input box styling */
    .stTextInput input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #2c5aa0;
        box-shadow: 0 0 0 2px rgba(44, 90, 160, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #2c5aa0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #1e3f70;
        box-shadow: 0 4px 12px rgba(44, 90, 160, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #999;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# PYDANTIC MODEL FOR OUTPUT
# ============================================
class PolicyAnswer(BaseModel):
    """Output structure for answering the user's question."""
    answer: str = Field(description="A complete, helpful answer to the user's question based on the policy documents.")

# ============================================
# INITIALIZE RAG COMPONENTS
# ============================================
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components (cached for performance)"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please add it to your .env file or Streamlit secrets.")
        st.stop()
    
    # Set API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load vector store
    CHROMA_DB_PATH = "dry_clean_db"
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings_model
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Setup parser
    parser = JsonOutputParser(pydantic_object=PolicyAnswer)
    
    # System prompt
    SYSTEM_TEMPLATE = """
You are a helpful Dry Cleaning Policy Assistant. Your task is to answer the user's question accurately using the provided context from the policy documents.

IMPORTANT: The context contains patterns in this format: `r"question pattern answer text"`
The text AFTER the question pattern is the ANSWER you should provide.

Examples of how to read the patterns:
- `r"how much for a suit cost of suit for dry cleaning is $17 dollars"` 
  → Answer: "Suits cost $17 for dry cleaning"
  
- `r"when are you open Monday-Friday 7am-6pm and Saturdays 10am-8pm"`
  → Answer: "We're open Monday-Friday 7am-6pm and Saturdays 10am-8pm"

- `r"do you do leather yes we do clean leather jackets"`
  → Answer: "Yes, we clean leather jackets"

Instructions:
1. Carefully look through ALL the context provided for relevant information.
2. Extract the actual answer from the pattern format and provide it clearly to the user.
3. If the information is truly not in ANY of the context, respond with: "I don't have that information in the current policies."
4. Be concise but include all relevant details.
5. CRITICAL: Your response MUST ALWAYS be valid JSON in this exact format: {format_instructions}
6. NEVER respond with plain text. ALWAYS use the JSON format, even when information is not available.

Context:
{context}
"""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "{question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    # Create RAG chain
    rag_chain = prompt | llm | parser
    
    return vector_store, rag_chain

# ============================================
# QUERY FUNCTION
# ============================================
def get_answer(question: str, vector_store, rag_chain):
    """Get answer from the RAG system"""
    try:
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(question, k=5)
        
        # Combine context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        response_dict = rag_chain.invoke({"context": context_text, "question": question})
        
        # Extract answer
        answer = response_dict.get('answer', 'Sorry, I encountered an error processing your question.')
        
        return answer
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧼 Dry Cleaning Policy Assistant</h1>
        <p>Ask me anything about our dry cleaning policies, pricing, and services</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    try:
        vector_store, rag_chain = initialize_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize the system: {str(e)}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm here to help you with questions about our dry cleaning policies. You can ask me about pricing, hours, services, and more!"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about our policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                response = get_answer(prompt, vector_store, rag_chain)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("""
    <div class="footer">
        Powered by AI | Questions? Contact us at support@example.com
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with info (optional)
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        This AI assistant helps answer questions about our dry cleaning policies.
        
        **Common Questions:**
        - Pricing information
        - Hours of operation  
        - Services offered
        - Pickup & delivery
        
        **Tip:** Ask natural questions like "How much for a suit?" or "When are you open?"
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
