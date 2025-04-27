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
load_dotenv()

loader=PyPDFDirectoryLoader("D:\Interview\SQL-Assistant-Experiments\data")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    is_separator_regex=False,
)

documents=text_splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore=FAISS.from_documents(documents=documents, embedding=embeddings)

retriever=vectorstore.as_retriever()


prompt=ChatPromptTemplate.from_template(
    '''
You are an expert backend database engineer working in a healthcare IT system. 
Your role is to process service requests (SRs) raised by users who are unable to make changes from the UI or need reports for decision-making.

Only answer to the questions which are exists in the database schema only. If you get any unknown table or column request then do not respond.

These requests relate to backend database operations such as:
- Correcting wrong data entries (e.g., date of birth, phone number, name)
- Updating missing or outdated values
- Generating reports or summaries from existing data
- Assigning entities (e.g., doctors to hospitals)
- Reviewing logs of appointments or patient records

You have full knowledge of the database schema, which includes tables like:
- Doctors, Doctor_Licenses, Specializations, Doctor_Specializations
- Hospitals, Departments, Doctor_Hospital_Assignments
- Patients, Appointments

### Your task:
Given a natural language service request, understand the user's intent and generate:
1. The *Action Type* (e.g., Update, Correction, Report)
2. The *Target Table(s)*
3. The *SQL Query* or stored procedure to resolve the issue
4. A short *explanation* of what the query does

Make sure the output is aligned with schema column names and formatted for clarity.

---
{context}
### Input:
{input}

### Output:
*Action Type:*  
*Target Table(s):*  
*SQL Query:*  
```sql
-- SQL query here
'''
)



llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=os.getenv("GEMINI_API_KEY"))

document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
retrieval_chain=create_retrieval_chain(retriever,document_chain)

st.title("Service Requests ChatBot")

with st.sidebar:
    st.markdown(
        '''
Sample Service Requests (Queries) for Healthcare Database*

1. *Update Request*
   - "Please update the phone number for patient ID 1042 to +91-8885544332."
   - "Change the email address for doctor with ID 210 to dr.ravi@hospital.com."
   - "Set the expiry_date for license ID 502 to 2027-05-31."

2. *Correction Request*
   - "Correct the date of birth for patient ID 1090 to 1992-10-05."
   - "The gender field for doctor ID 255 is incorrect; please set it to 'Female'."
   - "The name of the department with ID 15 should be 'Cardiology'. Please correct it."

3. *Assignment Request*
   - "Assign doctor ID 212 to hospital ID 4, department ID 3, starting from 2024-05-01."
   - "End the hospital assignment for doctor ID 208 in department ID 7."

4. *Report Request*
   - "Generate a report of all appointments for patient ID 1089 in April 2024."
   - "List all doctors specialized in Neurology currently assigned to hospital ID 3."
   - "Show all expired licenses for doctors as of today."
   - "Get the number of patients registered in hospital ID 5 last month."

5. *Bulk/Data Integrity*
   - "Update all phone numbers starting with 99999 to 88888 in the Patients table."
   - "List all departments on the 3rd floor in hospital ID 2."
   - "Show all appointments scheduled for next week across all hospitals."
'''
    )

input_text=st.text_input("Enter you query related to service request")

if input_text:
    response=retrieval_chain.invoke(
        {"input":input_text}
    )

    st.success(response['answer'])



