import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import pytesseract
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import StorageContext
from llama_index.core import Settings
import base64
import datetime

# Disable OpenAI LLM globally
Settings.llm = None

# Set environment variables for offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Explicitly set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\KH453QU\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load question-answering model
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)

# Initialize the pipeline
qa_model = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# Custom embedding class
class SentenceEmbedding(BaseEmbedding):
    model: SentenceTransformer

    def embed(self, texts: List[str]):
        return self.model.encode(texts).astype('float32')

    def _get_text_embedding(self, text: str):
        return self.embed([text])[0]

    def _aget_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

# Initialize custom embedding model
embedding_model = SentenceEmbedding(model=SentenceTransformer(r'C:\Users\KH453QU\OneDrive - EY\Desktop\MPSEDC Project\e-office\paraphrase-MiniLM-L6-v2'))
Settings.embed_model = embedding_model

# Function to extract text from PDF using Tesseract OCR
def extract_text_from_pdf(pdf_path):
    try:
        # Convert the PDF to images
        images = convert_from_path(pdf_path, poppler_path=r'C:\poppler-24.08.0\Library\bin')

        if len(images) == 0:
            return "Error: No pages were detected in the PDF."

        ocr_text = ""
        for i, image in enumerate(images):
            processed_image = preprocess_image(image)
            text = pytesseract.image_to_string(processed_image, lang='eng+hin', config='--oem 3 --psm 6')
            ocr_text += f"--- Page {i + 1} ---\n{text}\n"

        return ocr_text.strip()
    except Exception as e:
        return f"Error occurred: {e}"

# Function to preprocess the image for better OCR results
def preprocess_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return thresh_image

# Function to chunk text and create documents, including metadata with PDF file info
def create_documents_from_text(text, source_pdf, chunk_size=200):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(Document(text=' '.join(current_chunk), metadata={"source": source_pdf}))
            current_chunk = []

    if current_chunk:
        chunks.append(Document(text=' '.join(current_chunk), metadata={"source": source_pdf}))

    return chunks

# Function to setup the vector store
def setup_vector_store(documents: List[Document]):
    dimension = 384  
    faiss_index = faiss.IndexFlatL2(dimension)
    
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    nodes = []
    for doc in documents:
        embedding = embedding_model.embed([doc.text])[0]
        
        if embedding.shape[0] != dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected dimension {dimension}.")
            
        node = TextNode(
            text=doc.text,
            embedding=embedding,
            id_=str(hash(doc.text)),
            metadata=doc.metadata  # Include metadata with source PDF
        )
        nodes.append(node)
    
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embedding_model,
        llm=None
    )
    
    return index, [doc.text for doc in documents]

# Background image function
def add_bg_from_local(image_file, opacity=0):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url(data:image/jfif;base64,{encoded_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #002F74;
            color: white;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}
        .footer p {{
            font-style: italic;
            font-size: 14px;
            margin: 0;
            flex: 1 1 50%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_bg_from_local('grey_bg.jfif')

st.title("E - office AI Query Bot")

# Define the folder where all PDFs are located
pdf_folder = r'C:\Users\KH453QU\OneDrive - EY\Desktop\MPSEDC Project\e-office\Office Notice_Orders'

# Process all PDFs in the folder
documents = []
for pdf_filename in os.listdir(pdf_folder):
    if pdf_filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if pdf_text:
            pdf_documents = create_documents_from_text(pdf_text, source_pdf=pdf_filename)
            documents.extend(pdf_documents)

# If no documents found, show an error
if len(documents) == 0:
    st.error("No valid PDFs were found or the extracted text is empty.")
else:
    vector_index, node_texts = setup_vector_store(documents)

    query = st.text_input("Ask a question:")

    if st.button("Submit"):
        if query:
            with st.spinner("Searching..."):
                query_engine = vector_index.as_query_engine(llm=None)
                response = query_engine.query(query)

                if response and hasattr(response, 'nodes') and len(response.nodes) > 0:
                    context_with_source = ""
                    for node in response.nodes:
                        context_with_source += f"Source: {node.metadata['source']}\nText: {node.text}\n\n"

                    if not context_with_source.strip():
                        answer = "No relevant information found in the retrieved context."
                    else:
                        try:
                            model_response = qa_model(question=query, context=context_with_source)
                            answer = model_response['answer'] if 'answer' in model_response else "Answer not found."
                        except Exception as e:
                            answer = f"Error in generating answer: {e}"
                else:
                    answer = "No relevant information found."

            st.write("### Answer:")
            st.write(answer)

            feedback = st.radio("Was this answer helpful?", ("Yes", "No"), index=0)
            if feedback == "No":
                st.error("Thank you for your feedback! We will work on improving our system.")
            else:
                st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter a question.")

# Footer
current_year = datetime.datetime.now().year
footer = f"""
    <div class="footer">
        <p style="text-align: left;">Copyright © {current_year} MPSeDC. All rights reserved.</p>
        <p style="text-align: left;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
