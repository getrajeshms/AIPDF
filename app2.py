import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
import time
from io import BytesIO

# LangChain impor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datetime import datetime


# Custom CSS for modern styling
def display_header_with_logo(logo_path=None):
    """Display header with local logo using Streamlit's image handling"""
    # Create a container for the header with logo
    header_container = st.container()
    
    with header_container:
        # Create columns for logo and title
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-header-text">
                <h1 style="margin-bottom: 0.5rem; color: white; font-size: 2.5rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📚 AI PDF Research Assistant</h1>
                <p style="margin: 0.5rem 0; color: white; font-size: 1.2rem; opacity: 0.9;">Institute of Public Health, Bengaluru</p>
                <small style="color: white; opacity: 0.8;">Strengthening Health Systems | Celebrating 20 Years of Excellence</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Display logo with proper dimensions
            if logo_path and os.path.exists(logo_path):
                try:
                    st.image(logo_path, width=400)
                except Exception as e:
                    st.warning(f"⚠️ Logo image not found: {str(e)}")
            else:
                # Try to load logo from current directory with common names
                logo_files = ["logo.png", "IPH logo1.jpg", "iph_logo.png", "logo.jpg"]
                logo_loaded = False
                
                for logo_file in logo_files:
                    if os.path.exists(logo_file):
                        try:
                            st.image(logo_file, width=400)
                            logo_loaded = True
                            break
                        except Exception:
                            continue
                
                if not logo_loaded:
                    # Display placeholder with instructions
                    st.markdown("""
                    <div style="background: rgba(255,255,255,0.2); padding: 2rem; border-radius: 10px; text-align: center; color: white; border: 2px dashed rgba(255,255,255,0.5);">
                        <p><strong>📷 Logo Placeholder</strong></p>
                        <small>Place your logo file as 'logo.png' or 'IPH logo1.jpg' in the same directory</small>
                    </div>
                    """, unsafe_allow_html=True)


def load_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom header styling with background */
    .main > div:first-child {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 50%, #32CD32 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(46, 139, 87, 0.3);
    }
    
    .custom-header-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        padding-right: 1rem;
    }
    
    /* Ensure header background covers the container */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f0f8ff;
        border: 2px dashed #4169e1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #2E8B57, #228B22);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4);
        background: linear-gradient(45deg, #228B22, #32CD32);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message.user { 
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        margin-left: 20%;
    }
    
    .chat-message.bot { 
        background: linear-gradient(135deg, #4682B4 0%, #1E90FF 100%);
        margin-right: 20%;
    }
    
    .chat-message .avatar { 
        width: 60px;
        height: 60px;
        border-radius: 50%;
        overflow: hidden;
        margin-right: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .chat-message .avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .chat-message .message {
        flex: 1;
        color: white;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2E8B57;
    }
    
    /* Success/Warning/Error styling */
    .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E8B57;
        box-shadow: 0 0 0 3px rgba(46, 139, 87, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E8B57 0%, #228B22 100%);
    }
    
    /* File details styling */
    .file-details {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    /* Logo container styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .custom-header-text h1 {
            font-size: 1.8rem;
        }
        .custom-header-text p {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def display_file_details(pdf_docs):
    """Display uploaded file details in an attractive format"""
    if pdf_docs:
        st.markdown("### 📁 Uploaded Files")
        
        total_size = 0
        file_info = []
        
        for i, pdf in enumerate(pdf_docs):
            file_size = len(pdf.getvalue()) / 1024 / 1024  # Size in MB
            total_size += file_size
            file_info.append({
                'name': pdf.name,
                'size': f"{file_size:.2f} MB",
                'type': pdf.type
            })
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Files", len(pdf_docs))
        with col2:
            st.metric("💾 Total Size", f"{total_size:.2f} MB")
        with col3:
            st.metric("📊 Status", "Ready")
        
        # Display individual files
        for i, info in enumerate(file_info):
            st.markdown(f"""
            <div class="file-details">
                <strong>📄 {info['name']}</strong><br>
                <small>Size: {info['size']} | Type: {info['type']}</small>
            </div>
            """, unsafe_allow_html=True)


def get_pdf_text(pdf_docs):
    """Extract text from PDF files with progress tracking"""
    text = ""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf in enumerate(pdf_docs):
        status_text.text(f'Processing {pdf.name}...')
        try:
            pdf_reader = PdfReader(pdf)
            file_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                file_text += page.extract_text()
            text += file_text
            progress_bar.progress((i + 1) / len(pdf_docs))
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    
    status_text.text('✅ Text extraction complete!')
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()
    
    return text


def get_text_chunks(text, model_name):
    """Split text into chunks with improved parameters"""
    chunk_sizes = {
        "Gemini": 8000,
        "OpenAI": 4000,
        "Perplexity": 6000
    }
    
    chunk_size = chunk_sizes.get(model_name, 4000)
    overlap = min(1000, chunk_size // 8)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, model_name, api_key=None):
    """Create vector store with progress tracking"""
    embeddings = None
    if model_name == "Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )
    elif model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if embeddings is None:
        raise ValueError(f"Model '{model_name}' is not supported yet for embeddings.")

    with st.spinner(f"Creating embeddings for {len(text_chunks)} chunks..."):
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    
    return vector_store


def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    """Create conversational chain with improved prompt"""
    prompt_template = """
    You are a helpful AI assistant specialized in analyzing PDF documents. Please answer the question based on the provided context.
    
    Instructions:
    1. Provide a comprehensive and well-structured answer
    2. Use bullet points or numbered lists when appropriate
    3. If the answer is not in the context, clearly state that
    4. Cite specific sections or pages when possible
    5. Be conversational and helpful in your tone
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    if model_name == "Gemini":
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key
        )
    elif model_name == "OpenAI":
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
    else:
        raise ValueError(f"Model '{model_name}' not implemented yet for QA chain.")

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def display_chat_message(question, answer, is_user=True):
    """Display chat messages with modern styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="User">
            </div>
            <div class="message">{question}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" alt="Bot">
            </div>
            <div class="message">{answer}</div>
        </div>
        """, unsafe_allow_html=True)


def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    """Process user input and generate response"""
    if not api_key:
        st.warning("⚠️ Please provide your API key to continue.")
        return
    
    if not pdf_docs:
        st.warning("📁 Please upload PDF files before asking questions.")
        return

    with st.spinner("🤖 Processing your question..."):
        try:
            # Extract text and create chunks
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text.strip():
                st.error("❌ No text could be extracted from the uploaded PDFs.")
                return
            
            text_chunks = get_text_chunks(raw_text, model_name)
            vector_store = get_vector_store(text_chunks, model_name, api_key)
            
            # Set up embeddings for similarity search
            if model_name == "Gemini":
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", google_api_key=api_key
                )
            elif model_name == "OpenAI":
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # Perform similarity search and get response
            new_db = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
            docs = new_db.similarity_search(user_question, k=4)
            chain = get_conversational_chain(model_name, vectorstore=new_db, api_key=api_key)
            
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True,
            )
            
            answer = response["output_text"]
            
            # Display chat messages
            display_chat_message(user_question, "", is_user=True)
            display_chat_message("", answer, is_user=False)
            
            # Save to conversation history
            pdf_names = [pdf.name for pdf in pdf_docs]
            conversation_history.append((
                user_question,
                answer,
                model_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(pdf_names),
            ))
            
            st.success("✅ Response generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")


def export_conversation_history(conversation_history):
    """Export conversation history with improved formatting"""
    if conversation_history:
        df = pd.DataFrame(
            conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Files"]
        )
        
        # Create download button with custom styling
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        st.markdown("### 📥 Export Options")
        st.download_button(
            label="📊 Download Conversation History (CSV)",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def main():
    """Main application function"""
    st.set_page_config(
        page_title="IPH PDF Research Assistant", 
        page_icon="📚", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Define logo path - Update this path to your local image
    logo_path = "logo.png"  # Put your logo file here
    
    # Display header with logo
    display_header_with_logo(logo_path)
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "question_submitted" not in st.session_state:
        st.session_state.question_submitted = False
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection with descriptions
        model_descriptions = {
            "Gemini": "🧠 Google's powerful language model",
            "OpenAI": "🚀 GPT-4 powered responses",
            
        }
        
        model_name = st.radio(
            "Select AI Model:",
            options=list(model_descriptions.keys()),
            format_func=lambda x: model_descriptions[x]
        )
        
        # API Key input with validation
        api_key = st.text_input(
            f"🔑 Enter your {model_name} API Key:",
            type="password",
            help=f"Get your API key from the {model_name} platform"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required")
        
        st.divider()
        
        # Control buttons
        st.markdown("### 🎛️ Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.question_submitted = False
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if st.session_state.conversation_history:
                    st.session_state.conversation_history = []
                    st.session_state.question_submitted = False
                    st.success("Chat cleared!")
        
        st.divider()
        
        # File upload section
        st.markdown("### 📁 Upload Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to chat with"
        )
        
        if pdf_docs:
            display_file_details(pdf_docs)
            
            if st.button("🚀 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        # Test text extraction
                        test_text = get_pdf_text(pdf_docs[:1])
                        if test_text.strip():
                            st.success("✅ Documents processed successfully!")
                        else:
                            st.warning("⚠️ No text found in documents")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Export section
        if st.session_state.conversation_history:
            st.divider()
            export_conversation_history(st.session_state.conversation_history)
    
    # Main chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("#### Previous Conversations")
        for question, answer, model, timestamp, pdf_files in st.session_state.conversation_history[-5:]:
            display_chat_message(question, "", is_user=True)
            display_chat_message("", answer, is_user=False)
            st.caption(f"🕐 {timestamp} | 🤖 {model} | 📁 {pdf_files}")
            st.divider()
    
    # Question input form - Fixed to prevent loop
    with st.form("question_form"):
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about your PDFs?",
            help="Type your question and click Submit"
        )
        
        submitted = st.form_submit_button("🚀 Submit Question", use_container_width=True)
    
    # Process question only when form is submitted
    if submitted and user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <small>
                📚 IPH PDF Research Assistant | Institute of Public Health, Bengaluru | 
                <a href='#' style='color: #2E8B57;'>DSIR Recognized SIRO</a> | 
                <a href='#' style='color: #2E8B57;'>20 Years of Excellence</a>
            </small>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()