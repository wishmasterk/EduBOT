import os
import shutil
import tempfile
import easyocr
import numpy as np
import streamlit as st
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from dotenv import load_dotenv
from easyocr import Reader
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
poppler_path = os.getenv('POPPLER_PATH')
USER_PDF_DIRECTORY = "USER_PDFs"

# STEP 1: UPLOAD AND LOAD RAW PDF(s)
# UPLOAD MULTIPLE PDFS
def upload_pdfs(uploaded_files):
    """Save multiple PDF files to disk with sanitized filenames"""
    saved_paths = []

    try:
        # Clean any existing files in USER_PDF_DIRECTORY directory
        if os.path.exists(USER_PDF_DIRECTORY):
            shutil.rmtree(USER_PDF_DIRECTORY)
        os.makedirs(USER_PDF_DIRECTORY)
        
        for uploaded_file in uploaded_files:
            try:
                # Sanitize filename
                safe_name = Path(uploaded_file.name).name
                file_path = os.path.join(USER_PDF_DIRECTORY, safe_name)

                # Write file
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

            except Exception as e:
                st.error(f"❌ Failed to save {uploaded_file.name}: {str(e)}!")
                continue

        if not saved_paths:
            st.error("❌ No files were successfully uploaded!")

        return saved_paths

    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}!")
        # Clean up temp directory in case of error
        if os.path.exists(USER_PDF_DIRECTORY):
            shutil.rmtree(USER_PDF_DIRECTORY)
        raise e

#Cleanup function that can be called when processing is complete
def cleanup_temp_pdfs():
    """Clean up the PDF directory"""
    if os.path.exists(USER_PDF_DIRECTORY):
        try:
            shutil.rmtree(USER_PDF_DIRECTORY)
        except Exception as e:
            st.error(f"❌ Failed to clean up temporary files: {str(e)}!")

# FOR PDFs CONTAINING TEXT IN IMAGES
def OCR_pdf(pdf_path):
    """Extract text from image-based PDF using OCR"""
    try:
        reader = easyocr.Reader(['en'])  # Initialize once and reuse

        # Convert PDF to images
        images = convert_from_path(pdf_path, poppler_path = poppler_path)
        if not images:
            return None

        extracted_text = []
        for i, image in enumerate(images):
            # Convert PIL Image to numpy array for EasyOCR
            image_np = np.array(image)
            
            # Perform OCR on each image
            results = reader.readtext(image_np)
            
            # Extract text from results
            page_text = []
            for (bbox, text, prob) in results:
                if text.strip() and prob > 0.5:  # Only add text with confidence > 50%
                    page_text.append(text)
            
            if page_text:
                page_content = " ".join(page_text)
                extracted_text.append(page_content)

        if not extracted_text:
            return None
            
        # Return as a list with one item (combined text)
        return ["\n\n".join(extracted_text)]

    except Exception as e:
        print(e)
        return None
    
# STEP 2: LOAD MULTIPLE PDFS (Parallel Processing)
def load_pdfs(file_paths):
    """Load and validate multiple PDFs with parallel processing"""
    pdf_documents = {}  # Dictionary to store documents for each PDF = []
    failed_files = []

    for file_path in file_paths:
        try:
            pdf_name = os.path.basename(file_path)
            # First attempt: Regular PDF text extraction
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()

            if not file_docs: # empty list for the pdfs whose pages cannot be extracted
                pdf_documents[pdf_name] = []
                continue

             # Check if any page has content
            has_content = any(doc.page_content.strip() for doc in file_docs)
            
            if not has_content:
                # Second attempt: OCR if no text was extracted
                ocr_result = OCR_pdf(file_path)
                
                if ocr_result:
                    # Create Document objects from OCR text
                    ocr_docs = []
                    for i, text in enumerate(ocr_result):
                        if text.strip():
                            ocr_docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_name,
                                    "page": i + 1,
                                    "extraction_method": "OCR"
                                }
                            ))
                    if ocr_docs:
                        pdf_documents[pdf_name] = ocr_docs
                    else:
                        pdf_documents[pdf_name] = []
                        failed_files.append(pdf_name)

                else:
                    pdf_documents[pdf_name] = []
                    failed_files.append(os.path.basename(file_path))
            else:
                # Regular extraction succeeded
                for doc in file_docs:
                    doc.metadata["extraction_method"] = "PyPDFLoader"
                pdf_documents[pdf_name] = file_docs
                
        except Exception as e:
            st.error(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}!")
            pdf_documents[pdf_name] = []
            failed_files.append(os.path.basename(file_path))
            continue
    
    return pdf_documents # dictionary of pdf_name: list of documents

# STEP 2: CREATE CHUNKS FORM THE EXTRACTED TEXT
def create_chunks(pdf_documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        all_chunks = []
        failed_pdfs = []
        valid_files = 0

        for pdf_name, docs in pdf_documents.items():
            try:
                if docs == []: # for pdfs whose pages cannot be extracted
                    failed_pdfs.append(pdf_name)
                    continue

                pdf_chunks = text_splitter.split_documents(docs)    
                if pdf_chunks:
                    valid_files += 1
                    # Add PDF name to metadata of each chunk
                    for chunk in pdf_chunks:
                        chunk.metadata["pdf_name"] = pdf_name
                    
                    all_chunks.extend(pdf_chunks)
                else:
                    failed_pdfs.append(pdf_name)

            except Exception as e:
                st.error(f"❌ Error chunking {pdf_name}: {str(e)}!")
                failed_pdfs.append(pdf_name)
                continue
        
        if failed_pdfs:
            st.error(f"❌ Processing failed for the file(s): {', '.join(failed_pdfs)}")
        
        return all_chunks, valid_files
        
    except Exception as e:
        st.error(f"❌ Error during chunking process: {str(e)}!")
        return None, 0

# STEP 3: CREATE VECTOR EMBEDDINGS AND STORE THEM IN FAISS VECTOR DATABASE
def get_embedding_model():
    # model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions, model by Microsoft
    model = OpenAIEmbeddings(model = "text-embedding-3-small") # 3072 dimensions, model by openai
    return model

def emd_and_store(text_chunks, model, FAISS_DB_PATH):
    """Add documents to existing FAISS store or create new one if none exists"""
    try:
        # Try to load existing FAISS store
        db = FAISS.load_local(FAISS_DB_PATH, model, allow_dangerous_deserialization=True)
        db.add_documents(text_chunks)
        st.info("✅ Added new PDF(s) to existing ones!")
    except:
        # If no existing store, create a new one
        db = FAISS.from_documents(text_chunks, model)
        st.success("✅ File(s) successfully stored!")

    db.save_local(FAISS_DB_PATH)




