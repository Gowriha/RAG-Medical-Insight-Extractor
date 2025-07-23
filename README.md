# RAG-Medical-Report-Assisstant
Simplifying patient medical records using Retrieval-Augmented Generation (RAG) and Google Gemini.

## Overview

This project leverages RAG with Google Generative AI to help users understand complex medical documents (PDFs). It extracts relevant context, embeds it using FAISS, and answers user questions in simple, accurate medical language.


## Features

- Upload & process medical PDFs  
- Ask natural language questions
- Multilingual Support
- Downloadable Insights
- FAISS vector store for fast retrieval  
- Streamlit UI with a clean layout  

## Tech Stack

| Layer         | Technology                |
|---------------|---------------------------|
| Frontend      | Streamlit                 |
| Backend       | Python + LangChain        |
| Embedding     | `embedding-001` (Gemini)  |
| LLM           | `gemini-pro`              |
| Vector Store  | FAISS                     |
| File Handling | PyPDF2                    |

---

## Installation

1. Clone the repository  
git clone  
cd medical-report-assistant  
2. Install dependencies  
pip install -r requirements.txt
3. Run the app  
streamlit run app.py

# Output
![WhatsApp Image 2025-07-23 at 5 26 29 PM](https://github.com/user-attachments/assets/9d29d980-f5c7-448c-9931-74f482a725fe)
![WhatsApp Image 2025-07-23 at 5 07 58 PM](https://github.com/user-attachments/assets/545ec751-e070-489b-bfbd-af94558709c5)
![WhatsApp Image 2025-07-23 at 5 07 23 PM](https://github.com/user-attachments/assets/a9a69ee7-9664-41a8-b256-c65e196cccff)
