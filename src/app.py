from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables and set page configuration
load_dotenv()
st.set_page_config(
    page_title="Healthcare DB Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
    }
    .subheader {
        font-size: 18px;
        font-weight: 500;
        color: #34495e;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .success-response {
        background-color: #eaf7ea;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1 class='main-header'>Healthcare Database Request Assistant 🏥</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Process database service requests for healthcare professionals</p>", unsafe_allow_html=True)

# Function definitions
def load_documents(directory_path):
    try:
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def process_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(docs)
    return documents

def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU usage to avoid CUDA/meta tensor issues
            encode_kwargs={'normalize_embeddings': True}  # Ensure embeddings are normalized
            )
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

def setup_llm_chain():
    prompt = ChatPromptTemplate.from_template(
        '''
You are HDBRA (Healthcare Database Request Assistant), an expert backend database engineer specializing in healthcare IT systems.
Your primary responsibility is to process service requests (SRs) from healthcare professionals who need database-level changes or reports that cannot be performed through the user interface.

### Database Context:
You have access to a comprehensive healthcare database with the following schema:
- Doctors (doctor_id, name, gender, email, phone, date_joined, status)
- Doctor_Licenses (license_id, doctor_id, license_number, issue_date, expiry_date, issuing_authority)
- Specializations (specialization_id, name, description)
- Doctor_Specializations (doctor_id, specialization_id, certification_date)
- Hospitals (hospital_id, name, address, city, state, contact_number, email)
- Departments (dept_id, hospital_id, name, floor, wing, head_doctor_id)
- Doctor_Hospital_Assignments (assignment_id, doctor_id, hospital_id, dept_id, start_date, end_date, status)
- Patients (patient_id, name, gender, date_of_birth, address, phone, email, blood_group, registration_date)
- Appointments (appointment_id, patient_id, doctor_id, hospital_id, dept_id, appointment_date, status, notes)

### Your task:
For each service request, provide a professional response with:
1. Action Type: Categorize the request (Update, Correction, Assignment, Report, Data Integrity)
2. Target Table(s): Identify the primary tables affected by the request
3. SQL Solution: Generate the precise SQL query or stored procedure to fulfill the request
4. Explanation: Provide a clear, concise explanation of what the SQL does and any considerations

### Rules:
- Only respond to service requests that exactly matches tables and columns listed in Database Context.
- If the request refers to any unknown table or column, then dont reply.
- Do NOT guess or assume any mappings. Only respond if the input clearly matches the schema.


Always prioritize data integrity and follow healthcare data handling best practices. For updates to critical fields, include appropriate WHERE clauses to ensure precise targeting.

{context}
### Service Request:
{input}

### Response:
Action Type: 
Target Table(s): 
SQL Solution:
sql
-- SQL query here

Explanation: 
'''
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return document_chain

# Sidebar configuration
with st.sidebar:
    st.image("D:\Interview\SQL-Assistant-Experiments\logo\hosp.jpg", width=100)
    st.title("Request Guide")
    
    # Tabs in sidebar for better organization
    tab1, tab2 = st.tabs(["Sample Requests", "About"])
    
    with tab1:
        st.subheader("Sample Service Requests")
        with st.expander("📝 Update Requests", expanded=False):
            st.markdown("""
            - Update the phone number for patient ID 1042 to +91-8885544332
            - Change the email address for doctor with ID 210 to dr.ravi@hospital.com
            - Set the expiry_date for license ID 502 to 2027-05-31
            """)
        
        with st.expander("🔄 Correction Requests", expanded=False):
            st.markdown("""
            - Correct the date of birth for patient ID 1090 to 1992-10-05
            - The gender field for doctor ID 255 is incorrect; please set it to 'Female'
            - The name of the department with ID 15 should be 'Cardiology'
            """)
            
        with st.expander("🔗 Assignment Requests", expanded=False):
            st.markdown("""
            - Assign doctor ID 212 to hospital ID 4, department ID 3, starting from 2024-05-01
            - End the hospital assignment for doctor ID 208 in department ID 7
            """)
            
        with st.expander("📊 Report Requests", expanded=False):
            st.markdown("""
            - Generate a report of all appointments for patient ID 1089 in April 2024
            - List all doctors specialized in Neurology currently assigned to hospital ID 3
            - Show all expired licenses for doctors as of today
            - Get the number of patients registered in hospital ID 5 last month
            """)
            
        with st.expander("🔍 Data Integrity", expanded=False):
            st.markdown("""
            - Update all phone numbers starting with 99999 to 88888 in the Patients table
            - List all departments on the 3rd floor in hospital ID 2
            - Show all appointments scheduled for next week across all hospitals
            """)
    
    with tab2:
        st.markdown("""
        ### About HDBRA
        
        This assistant helps healthcare professionals process database-level service requests that cannot be handled through the UI.
        
        *Features:*
        - SQL query generation
        - Data correction assistance
        - Report generation
        - Assignment management
        
        Built with LangChain and Gemini AI
        """)
    
    st.divider()
    st.caption("© 2025 Healthcare IT Services")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='query-box'>", unsafe_allow_html=True)
    input_text = st.text_area("Enter your service request:", height=100, 
                            placeholder="Example: Update the phone number for patient ID 1042 to +91-8885544332")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        submit_button = st.button("Process Request", use_container_width=True)
    with col_btn2:
        clear_button = st.button("Clear", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Request Stats")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    st.markdown(f"*Date:* {date_str}")
    st.markdown(f"*Time:* {time_str}")
    
    # Simple stats display
    req_stats = {
        "Updates": len([h for h in st.session_state.history if "Update" in h[1] or "Correction" in h[1]]),
        "Reports": len([h for h in st.session_state.history if "Report" in h[1]]),
        "Assignments": len([h for h in st.session_state.history if "Assignment" in h[1]]),
        "Total": len(st.session_state.history)
    }
    
    # Convert stats to DataFrame for better display
    stats_df = pd.DataFrame(list(req_stats.items()), columns=["Request Type", "Count"])
    st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    export_button = st.button("Export History", use_container_width=True)
    if export_button and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history, columns=["Request", "Response"])
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"request_history_{date_str}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Clear input and response if clear button is clicked
if clear_button:
    st.rerun()

# Load and process documents
# In a production environment, you might want to cache this step
data_directory = "D:\Interview\SQL-Assistant-Experiments\data"  # Replace with your actual path
docs = load_documents(data_directory)

if docs:
    documents = process_documents(docs)
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    
    # Create and setup the chain
    document_chain = setup_llm_chain()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Process the request when the button is clicked
    if submit_button and input_text:
        with st.spinner("Processing your request..."):
            response = retrieval_chain.invoke({"input": input_text})
            
            # Extract action type for history
            action_type = "Unknown"
            if "Action Type:" in response['answer']:
                action_start = response['answer'].find("Action Type:") + len("Action Type:")
                action_end = response['answer'].find("\n", action_start)
                action_type = response['answer'][action_start:action_end].strip()
            
            # Add to history
            st.session_state.history.append((input_text, action_type))
            
            # Display the response with styling
            st.markdown("<div class='success-response'>", unsafe_allow_html=True)
            st.markdown("### Response:")
            st.markdown(response['answer'])
            st.markdown("</div>", unsafe_allow_html=True)

# Display history
if st.session_state.history:
    with st.expander("Request History", expanded=False):
        for i, (req, act) in enumerate(st.session_state.history):
            st.markdown(f"*Request {i+1}:* {req} (Action: {act})")
            st.divider()