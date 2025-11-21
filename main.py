import os
import io
import base64
import shutil
import librosa
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder  # For microphone input
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Any, List
from agent_architect import *
from rag import *

load_dotenv(override = True)

FAISS_DB_PATH = "FAISS_VECTORSTORE"
st.set_page_config(layout = "centered", page_title = "EduBOT", initial_sidebar_state = "expanded")

# Initialize session state
if 'init' not in st.session_state:
    st.session_state.files_processed = False
    st.session_state.selected_model = None
    st.session_state.last_model = None # only for a problem related with audio mode
    st.session_state.input_mode = None
    st.session_state.messages = []
    st.session_state.openai_client = OpenAI()
    st.session_state.audio_bytes = None             # Stores raw audio data
    st.session_state.audio_query = ""               # Stores transcribed text
    st.session_state.user_query = None              # Stores user query
    st.session_state.text_query = ""                # Empty string for text query
    st.session_state.file_query = ""                # Empty string for content in files
    st.session_state.image_query = []               # Empty list for image query
    st.session_state.init = True

# Read and convert the image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Logo and header
logo_path = "edubot_logo.png"
logo_data_url = get_base64_image(logo_path)

# LOGO
st.markdown("###")
st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; margin-top: 30px;'> 
        <img src="{logo_data_url}" style="height: 100px; margin-right: -20px;" /> 
        <span style='font-size: 2.5em; font-weight: bold;'>EduBOT is <span style='color:#4CAF50;'>online</span>.</span>
    </div>
    <div style='text-align: center; font-size: 1.5em; margin-top: -10px; font-style: italic; color: #555;'>
        Feed me your curiosity — and I'll feed you insights.
    </div>
""", unsafe_allow_html=True)

st.markdown("###")

st.markdown("""
    <style>
        /* Sidebar title styling */
        .sidebar-title {
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
            margin-top: -70px;  /* Move title up */
            margin-bottom: 20px;
            border-bottom: 2px solid #4CAF50;
        }

        /* Style for EDU */
        .edu-text {
            color: #333;
        }

        /* Style for BOT */
        .bot-text {
            color: #4CAF50;
        }

        /* Center the sidebar content */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0;
        }

        /* Style for radio buttons */
        .stRadio > label {
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        
        /* Style radio button container */
        .stRadio > div {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 0px;  /* Adjust spacing after title */
            margin-bottom: -10px;
        }

        /* Style individual radio buttons */
        .stRadio > div > div > label {
            padding: 8px 15px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }

        .stRadio > div > div > label:hover {
            background-color: #e8f5e9;
        }

        /* Style the selectbox container */
        .stSelectbox {
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 15px 0;
            padding: 10px;
        }

        /* Style the selectbox label */
        .stSelectbox > label {
            color: #333;
            font-weight: 600;
            font-size: 16px;
        }

        /* Style the selectbox */
        .stSelectbox > div > div {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            transition: all 0.3s ease;
        }

        .stSelectbox > div > div:hover {
            border-color: #4CAF50;
        }

        /* Add spacing before upload section */
        .custom-uploader {
            margin-top: -10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:

    st.markdown("""
        <div class="sidebar-title">
            <span class="Edu-text">Edu</span><span class="bot-text">BOT</span>
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Input Mode",
        options=["🎤 Voice Assistant", "📝 Text & Files"],
        index=1,  # Default to Text & Files
        help="Choose mode in which you want to interact with the bot")
    
    # Store the selection in session state
    st.session_state.input_mode = input_mode

    # Add a small space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add selectbox for model selection
    option = st.selectbox(
        "Choose Response Mode",
        options=["Select a mode", "Search 🔍", "Reason 💡"],
        help="Select model based on your need, 🔍 for search, 💡 for reasoning.",
        key="model_selector"
    )

    # Update session state based on selection
    if option == "Select a mode":
        st.session_state.selected_model = None
    elif option == "Search 🔍":
        st.session_state.selected_model = "search"
    elif option == "Reason 💡":
        st.session_state.selected_model = "reason"
    
    # Add a small space
    st.markdown("<br>", unsafe_allow_html=True)

    # Your existing upload section
    st.markdown("""
        <style>
            .custom-uploader {
                background-color: #f5f8fb;
                border-radius: 20px;
                padding: 5px;
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                margin-bottom: 0px;
            }
            .custom-uploader:hover {
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            }
            .upload-icon {
                text-align: center;
                font-size: 36px;
                color: #8e44ad;
                margin-bottom: 10px;
            }
            .upload-title {
                text-align: center;
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .upload-desc {
                text-align: center;
                font-size: 14px;
                color: #555;
            }
        </style>

        <div class="custom-uploader">
            <div class="upload-icon">📚</div>
            <div class="upload-title">Upload Your File(s)</div>
            <p class="upload-desc">Choose file(s) to analyze</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(label = "box", type = ["pdf"], label_visibility = "hidden",
                                      accept_multiple_files = True,
                                      help = " ⚠️ Please upload valid **PDF(s)** only! "
                                            "The chatbot will analyze the text and answer questions based on your uploaded content. "
                                            "Once submitted, the files **cannot be removed**, and the model will be optimized based on the uploaded files"
                                            " (**Text extraction only**). Some PDFs can take a **little longer** to process.")

    if st.button("Submit"):
        if uploaded_files:
            try:
                with st.spinner("Processing..."):
                    files_path = upload_pdfs(uploaded_files)
                    documents = load_pdfs(files_path)
                    if not documents:
                        st.error("❌ PDF processing stopped!")
                        cleanup_temp_pdfs()
                        st.error("❌ Current execution stopped, try again with different file(s)!")
                        st.stop()  # Stop here if document loading failed
                    
                    text_chunks, valid_files = create_chunks(documents)
                    if not text_chunks:
                        st.error("❌ Chunking failed!")
                        cleanup_temp_pdfs()
                        st.error("❌ Current execution stopped, try again with different file(s)!")
                        st.stop()  # Stop here if chunking failed
                    
                    emd_and_store(text_chunks, get_embedding_model(), FAISS_DB_PATH)
                st.success(f"✅ {valid_files} file(s) successfully processed!")
                st.session_state.files_processed = True
                # Clean up temporary files after successful processing
                cleanup_temp_pdfs()
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}!")
                cleanup_temp_pdfs()  # Clean up even if processing fails
        else:
            st.warning("⚠️ Please upload at least a file before submitting!")

avatar_html = """
<style>
    .avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px;
        border: 2px solid #4CAF50;  /* Add a border */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* Add a subtle shadow */
    }
</style>
"""
# Display all messages
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = "edubot_avatar.jpg"
    else:
        avatar = "🔍" if st.session_state.selected_model == "search" else "💡"        
    with st.chat_message(message["role"], avatar = avatar):
        
        # Display PDF download buttons if present
        if "pdfs" in message and message["pdfs"]:
            st.write("📚 Attached Documents:")
            for pdf in message["pdfs"]:
                st.download_button(
                    label=f"Download {pdf['name']}",
                    data=pdf["file"],
                    file_name=pdf["name"],
                    mime="application/pdf"
                )

        # Display images if present
        if "images" in message and message["images"]:
            st.write("📸 Attached Images:")
            for img_info in message["images"]:
                # Display image
                image = Image.open(img_info["file"])
                st.image(image, caption=img_info["name"], width=300)  

        # Display audio if present
        if "audio" in message and message["audio"]:
            st.audio(message["audio"], format="audio/wav")

        # Display text if present
        if "text" in message and message["text"]:
            st.write(message["text"])


# Initialize Whisper model for speech-to-text
@st.cache_resource(ttl=3600, show_spinner=False)
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny") # use medium or large for better accuracy
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, model

processor, model = load_whisper_model()

def audio_to_text(audio_bytes):
# Create a temporary directory in the current working directory
    temp_dir = "temp_audio"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:
        # Use a specific file path in our temp directory
        temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
        
        # Write the audio bytes to the file
        with open(temp_file_path, "wb") as tmp:
            tmp.write(audio_bytes)
        
        # Process the audio
        audio_array, _ = librosa.load(temp_file_path, sr=16000)
        inputs = processor(audio_array, return_tensors="pt", sampling_rate=16000)
        predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Remove the directory along with the temp file
        shutil.rmtree(temp_dir)
        return transcription
        
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}!")
        shutil.rmtree(temp_dir)
        return None
    
    finally: # runs first
        # Try to clean up the temp file if it exists
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except:
            pass

@st.cache_data(ttl=3600, show_spinner=False)
def image_to_text(images: List[Any]) -> List[str]:
    """
    Convert one or more images to detailed text descriptions.

    Args:
        images: A list of file-like image objects (e.g., Streamlit UploadedFile).

    Returns:
        A list of strings, each formatted as:
        "[IMAGE: <filename>] <description>"
    """
    descriptions: List[str] = []
    prompt = "Describe this image in detail for document search"

    for img_file in images:
        # Load the image into a PIL Image
        image = Image.open(img_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call GPT-4o vision via your OpenAI client
        response = st.session_state.openai_client.chat.completions.create(
            model = "gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens = 1024,
        )

        desc = response.choices[0].message.content.strip()
        # Tag with image filename
        filename = getattr(img_file, "name", "unknown")
        descriptions.append(f"[IMAGE: {filename}] {desc}")
    return descriptions


if st.session_state.input_mode == "📝 Text & Files":
    st.session_state.user_query = st.chat_input(placeholder="Got a question? Fire away...",
                                                accept_file = "multiple",
                                                file_type = ["png", "jpg", "jpeg"]) 
    
    if st.session_state.user_query: # User query is a dictionary -> {"text": "...", "files": [...]}
        if st.session_state.files_processed and st.session_state.selected_model:
            
            # CASE 1: When input is text only
            if st.session_state.user_query["text"] != "" and st.session_state.user_query["files"] == []:
                st.session_state.text_query = st.session_state.user_query["text"].strip()

                st.session_state.messages.append({
                    "role": "user",
                    "type": "text",
                    "model": st.session_state.selected_model,
                    "text": st.session_state.text_query
                })

                with st.spinner("Thinking..."):
                    try:
                        response = agent.invoke({"messages": [{"role": "user" , "content": st.session_state.text_query}], "mode": st.session_state.selected_model}, 
                                                                config={"configurable": {"thread_id": "user123"}})["messages"][-1].content

                        # Add assistant's response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "model": st.session_state.selected_model,
                            "text": response
                        })

                    except Exception as e:
                        st.error(f"Error processing your request: {str(e)}")

                # Clear the input box after processing
                st.session_state.text_query = ""
                st.session_state.user_query = None
                st.rerun()

            # CASE 2: When input is can be any of the following:
            # 1. Text and files
            # 2. Files only
            elif (st.session_state.user_query["files"] != [] and st.session_state.user_query["text"] == "") or (st.session_state.user_query["files"] != [] and st.session_state.user_query["text"] != ""):
                
                st.session_state.text_query = st.session_state.user_query["text"].strip() if "text" in st.session_state.user_query else ""
                files = st.session_state.user_query["files"]

                images = []
                image_display_info = []
                
                for file in files:
                    file_type = file.type
                    if file_type in ["image/png", "image/jpeg", "image/jpg", "image/PNG", "image/JPEG", "image/JPG"]:
                        images.append(file)
                        image_display_info.append({
                            "name": file.name,
                            "file": file,
                            "type": "image"
                        })

                # Process Images
                if images:
                    try:
                        with st.spinner("Processing Images..."):
                            for img in images:
                                # Open and process each image
                                image = Image.open(img)
                                    
                                # Display the image
                                st.image(image, caption=f"Processing: {img.name}", width=300)
                                    
                                st.session_state.image_query.append(f"{img.name}")
                            
                        st.success(f"✅ Successfully processed {len(images)} image file(s)!")

                    except Exception as e:
                        st.error(f"❌ Error processing images: {str(e)}!")
                
                # Add to chat history                
                # Add as a single message
                st.session_state.messages.append({
                    "role": "user",
                    "type": "mixed",
                    "model": st.session_state.selected_model,
                    "images": image_display_info,  # These are the images that are displayed in the chat (list)
                    "text": st.session_state.text_query, # This is the text that is displayed in the chat (string)
                })  

                st.session_state.text_image_query = st.session_state.text_query + "\n\n" + "\n\n".join(image_to_text(files))

                with st.spinner("Thinking..."):
                    try:
                        response = agent.invoke({"messages": [{"role": "user" , "content": st.session_state.text_image_query}], "mode": st.session_state.selected_model}, 
                                                                config={"configurable": {"thread_id": "user123"}})["messages"][-1].content

                        # Add assistant's response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "model": st.session_state.selected_model,
                            "text": response
                        })

                        st.session_state.text_query = ""
                        st.session_state.image_query = []
                        st.session_state.user_query = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error processing your request: {str(e)}!")

        else:
            st.warning("⚠️ Please upload file(s) and select a response mode first!")            

# VOICE        
else:
    if st.session_state.files_processed:
        if st.session_state.selected_model:
            with st._bottom:
                _, _, column, _, _= st.columns([1, 1, 0.8, 1, 1])
                with column:
                    with st.container():
                        st.session_state.audio_bytes = audio_recorder(
                            pause_threshold = 3.0,  # Stop after 3 seconds of silence
                            text = "Record",
                            recording_color = "#e74c3c",  # Red when recording
                            neutral_color = "#2ecc71",  # Green when ready
                            icon_name = "microphone",  # Mic icon
                            key="voice_recorder",  #  unique key
                            icon_size = "4x",
                        )

            if (st.session_state.audio_bytes and not st.session_state.audio_query) and (st.session_state.selected_model == st.session_state.last_model):
                avatar = "🔍" if st.session_state.selected_model == "search" else "💡"
                with st.chat_message("user", avatar=avatar):
                    st.audio(st.session_state.audio_bytes, format="audio/wav")

                try:
                    # Convert audio to text
                    st.session_state.audio_query = audio_to_text(st.session_state.audio_bytes)

                    # Add to chat immediately
                    st.session_state.messages.append({
                        "role": "user",
                        "type": "audio",  # Mark as audio input
                        "model": st.session_state.selected_model,
                        "audio": st.session_state.audio_bytes
                    })

                    # Clear audio buffer (keep query text)
                    st.session_state.audio_bytes = None

                    with st.spinner("Thinking..."):
                        response = agent.invoke({"messages": [{"role": "user" , "content": st.session_state.audio_query}], "mode": st.session_state.selected_model}, 
                                                                config={"configurable": {"thread_id": "user123"}})["messages"][-1].content

                    # Add bot response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "model": st.session_state.selected_model,
                        "text": response
                    })

                    # Display bot response
                    avatar = "edubot_avatar.jpg"
                    with st.chat_message("assistant", avatar = avatar):
                        st.write(response)                    

                    st.session_state.audio_bytes = None
                    st.session_state.audio_query = ""

                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}!")
                    st.session_state.audio_bytes = None

            if st.session_state.selected_model != st.session_state.last_model:
                st.session_state.last_model = st.session_state.selected_model
                    
        else:
            st.warning("⚠️ Please select a response mode first!")
        
    else:
        st.warning("⚠️ Please upload at least a file for analysis!")


# with st.expander("Session State"):
#    st.json(st.session_state)
