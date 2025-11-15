# app_voice_attendance_enhanced_v2.py
import streamlit as st
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import json
import os
from datetime import datetime, date
import io
from pydub import AudioSegment
from scipy.spatial.distance import cosine
import librosa
# replaced pyttsx3 with gTTS
from gtts import gTTS
from io import BytesIO
import qrcode
from PIL import Image
import pandas as pd
from fpdf import FPDF
import base64
import urllib.parse
import time
from scipy.io import wavfile

# streamlit-webrtc for reliable microphone access
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Optional: firebase admin for backup
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# ----------------------------
# SAFE DEFAULTS (avoid NameError if Settings tab not opened)
# ----------------------------
speaker_threshold = 0.75
speaker_gap = 0.15
whisper_language = "en"
COUNTDOWN_SECONDS_DEFAULT = 5  # seconds per student (now max time)

# ----------------------------
# CONFIG / FILE PATHS
# ----------------------------
STUDENTS_FILE = "students_ai.json"
ATTENDANCE_FILE = "attendance_ai.json"
VOICE_EMBEDDINGS_DIR = "voice_embeddings"
os.makedirs(VOICE_EMBEDDINGS_DIR, exist_ok=True)

# ----------------------------
# SESSION STATE: Attendance flow variables
# ----------------------------
if "attendance_active" not in st.session_state:
    st.session_state.attendance_active = False
if "current_roll_index" not in st.session_state:
    st.session_state.current_roll_index = 0  # zero-based index into class student list
if "countdown" not in st.session_state:
    st.session_state.countdown = COUNTDOWN_SECONDS_DEFAULT
if "attendance_done" not in st.session_state:
    st.session_state.attendance_done = False
if "recording_started" not in st.session_state:
    st.session_state.recording_started = False
if "current_called" not in st.session_state:
    st.session_state.current_called = None
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "_attendance_order" not in st.session_state:
    st.session_state._attendance_order = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

# ----------------------------
# AI MODELS INITIALIZATION
# ----------------------------
@st.cache_resource
def load_ai_models():
    # Use tiny or base. For local Marathi/Hindi, consider using specific local models if base fails.
    whisper_model = whisper.load_model("base")  
    speaker_encoder = VoiceEncoder()
    return whisper_model, speaker_encoder

whisper_model, speaker_encoder = load_ai_models()

# ----------------------------
# TTS (gTTS - Streamlit safe)
# ----------------------------
def speak_text(text, lang="en"):
    """
    Generate TTS audio using gTTS and play it via st.audio.
    Note: gTTS requires internet access.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        # Play using Streamlit's audio player. This will play through system default output (Bluetooth if set).
        # Important: st.audio is non-blocking, so the countdown will start immediately.
        st.audio(buf, format="audio/mp3", autoplay=True) 
    except Exception as e:
        st.warning(f"TTS failed: {e}")

# ----------------------------
# LOAD / SAVE DATA
# ----------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

if "students" not in st.session_state:
    st.session_state.students = load_json(STUDENTS_FILE, {})
if "attendance" not in st.session_state:
    st.session_state.attendance = load_json(ATTENDANCE_FILE, {})

def save_data():
    save_json(STUDENTS_FILE, st.session_state.students)
    save_json(ATTENDANCE_FILE, st.session_state.attendance)

# ----------------------------
# FIREBASE BACKUP (optional)
# ----------------------------
FIREBASE_SERVICE_ACCOUNT = None
FIREBASE_COLLECTION = "attendance_backup"

def init_firebase(service_account_path):
    if not FIREBASE_AVAILABLE:
        st.warning("Firebase admin SDK not installed.")
        return None
    try:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        return db
    except Exception as e:
        st.error(f"Failed to init Firebase: {e}")
        return None

def backup_to_firebase(db):
    try:
        if db is None:
            st.info("Firebase DB not initialized.")
            return False
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "students": st.session_state.students,
            "attendance": st.session_state.attendance
        }
        db.collection(FIREBASE_COLLECTION).add(doc)
        return True
    except Exception as e:
        st.error(f"Firebase backup failed: {e}")
        return False

# ----------------------------
# AUDIO UTILITIES
# ----------------------------
def audio_bytes_to_wav(audio_bytes):
    """Convert audio bytes (webm/mpeg) to wav BytesIO (existing helper)."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        return wav_bytes
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return None

def save_np_audio_to_wav(np_audio, sample_rate, path):
    """
    Save numpy PCM array to WAV file (int16).
    np_audio: 1-D numpy float or int array
    sample_rate: e.g., 48000 or 16000
    """
    try:
        # Normalize if float
        if np_audio.dtype.kind == "f":
            # scale float32 [-1,1] -> int16
            maxv = np.max(np.abs(np_audio)) if np.max(np.abs(np_audio)) > 0 else 1.0
            scaled = (np_audio / maxv * 32767).astype(np.int16)
        else:
            scaled = np_audio.astype(np.int16)
        wavfile.write(path, sample_rate, scaled)
        return True
    except Exception as e:
        st.error(f"Failed to save WAV: {e}")
        return False

def transcribe_audio_whisper(audio_bytes=None, wav_path=None, language="en"):
    """Use Whisper AI to transcribe audio. Accepts raw bytes (webm) or a wav file path."""
    try:
        if wav_path:
            result = whisper_model.transcribe(wav_path, language=language, fp16=False)
        else:
            wav_bytes = audio_bytes_to_wav(audio_bytes)
            if wav_bytes is None:
                return None, 0.0
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(wav_bytes.read())
            result = whisper_model.transcribe(temp_path, language=language, fp16=False)
            os.remove(temp_path)

        text = result["text"].strip().lower()
        segs = result.get("segments", [])
        if segs:
            confidence = np.mean([seg.get("no_speech_prob", 0.0) for seg in segs])
            confidence = 1 - confidence
        else:
            confidence = 0.0
        return text, float(confidence)
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return None, 0.0

def create_voice_embedding(audio_bytes=None, wav_path=None):
    """Create embedding from audio bytes (webm) or wav path."""
    try:
        if wav_path:
            wav = preprocess_wav(wav_path)
            embedding = speaker_encoder.embed_utterance(wav)
            return embedding
        else:
            wav_bytes = audio_bytes_to_wav(audio_bytes)
            if wav_bytes is None:
                return None
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(wav_bytes.read())
            wav = preprocess_wav(temp_path)
            embedding = speaker_encoder.embed_utterance(wav)
            os.remove(temp_path)
            return embedding
    except Exception as e:
        st.error(f"Voice embedding failed: {e}")
        return None

def match_speaker(current_embedding, class_name):
    similarities = {}
    class_students = st.session_state.students.get(class_name, {})
    for student_name, info in class_students.items():
        if not info.get("voice_embedding"):
            continue
        saved_embedding = np.array(info["voice_embedding"])
        similarity = 1 - cosine(current_embedding, saved_embedding)
        similarities[student_name] = similarity
    if not similarities:
        return None, 0.0, 0.0
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    best_student, best_score = sorted_matches[0]
    second_best_score = sorted_matches[1][1] if len(sorted_matches) > 1 else 0.0
    return best_student, float(best_score), float(second_best_score)

def validate_audio_quality(audio_bytes):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        if audio.dBFS < -40:
            return False, "❌ Audio too quiet. Please speak louder!"
        if audio.dBFS > -10:
            return False, "❌ Audio too loud. Reduce volume!"
        duration = len(audio) / 1000.0
        if duration < 0.5:
            return False, "❌ Recording too short. Speak for at least 1 second."
        if duration > 10:
            return False, "❌ Recording too long. Keep it under 10 seconds."
        return True, "✅ Audio quality good"
    except Exception as e:
        return False, f"❌ Audio validation failed: {e}"

# ----------------------------
# QR GENERATION
# ----------------------------
def generate_qr_for_student(student_dict):
    data = json.dumps(student_dict)
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def get_image_download_link(img_pil, filename="qr.png"):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download QR</a>'
    return href

# ----------------------------
# APP LAYOUT / AUTH
# ----------------------------
st.set_page_config(page_title="🤖 AI Voice Attendance - Enhanced", layout="centered")
st.title("🤖 AI Voice Attendance System — Enhanced V2")
st.subheader("SIES HIGH SCHOOL MATUNGA")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_class" not in st.session_state:
    st.session_state.selected_class = None

if not st.session_state.logged_in:
    st.header("🔐 Teacher Login")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("👩‍🏫 Username")
        password = st.text_input("🔑 Password", type="password")
    with col2:
        std = st.selectbox("📘 Class (Standard)", [f"{i}" for i in range(1, 11)])
        division = st.selectbox("🏫 Division", list("ABCDEF"))
    if st.button("Login"):
        teacher_accounts = {"teacher1": "1234", "teacher2": "abcd"}
        if username in teacher_accounts and password == teacher_accounts[username]:
            st.session_state.logged_in = True
            st.session_state.selected_class = f"Class {std}{division}"
            st.success(f"✅ Logged in for {st.session_state.selected_class}")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.selected_class = None
    st.rerun()

class_name = st.session_state.selected_class
if class_name not in st.session_state.students:
    st.session_state.students[class_name] = {}
if class_name not in st.session_state.attendance:
    st.session_state.attendance[class_name] = []

# Sidebar summary
st.sidebar.header("🧑‍🏫 Menu")
st.sidebar.write(f"📘 Class: **{class_name}**")
st.sidebar.metric("👥 Students", len(st.session_state.students[class_name]))
today_str = date.today().strftime("%Y-%m-%d")
st.sidebar.metric("✅ Today's Attendance", len([a for a in st.session_state.attendance[class_name] if a.get("Date")==today_str]))

menu = st.sidebar.radio("Select Option:",
                        ["🎤 Take Attendance",
                         "➕ Add Student",
                         "📊 View Reports",
                         "👥 Registered Students",
                         "📈 Dashboard",
                         "⚙️ AI Settings",
                         "🔄 Backups"])

# ----------------------------
# AI SETTINGS (same as before)
# ----------------------------
if menu == "⚙️ AI Settings":
    st.subheader("⚙️ AI Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Speaker Recognition Thresholds**")
        speaker_threshold = st.slider("Minimum Confidence", 0.0, 1.0, speaker_threshold, 0.01)
        speaker_gap = st.slider("Min Gap from 2nd Best", 0.0, 0.5, speaker_gap, 0.01)
    with col2:
        whisper_language_choice = st.selectbox("Language", ["English", "Hindi", "Marathi"])
        language_map = {"English":"en","Hindi":"hi","Marathi":"mr"}
        whisper_language = language_map[whisper_language_choice]
    st.info(f"Speaker confidence ≥ {speaker_threshold*100:.0f}%, gap ≥ {speaker_gap*100:.0f}%, Language: {whisper_language_choice}")

# ----------------------------
# ADD STUDENT (same as before)
# ----------------------------
elif menu == "➕ Add Student":
    st.subheader(f"➕ Register New Student - {class_name}")
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Student Name*")
            roll_no = st.text_input("Roll Number*")
        with col2:
            bus_student = st.checkbox("🚌 Bus Student")
            local_student = st.checkbox("🏠 Local Student")
        photo = st.file_uploader("Upload Photo (optional)", type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Save Student Details")
    if submitted:
        if name and roll_no:
            student_entry = {
                "roll_no": roll_no,
                "bus": "Yes" if bus_student else "No",
                "local": "Yes" if local_student else "No",
                "voice_samples": [],
                "voice_embedding": None,
                "registered": False,
                "photo": None
            }
            if photo:
                image_bytes = photo.read()
                b64 = base64.b64encode(image_bytes).decode()
                student_entry["photo"] = b64
            st.session_state.students[class_name][name] = student_entry
            save_data()
            st.success(f"✅ Student {name} added! Now record voice samples.")
    st.divider()
    # Registration UI (same as before)
    if st.session_state.students[class_name]:
        selected_student = st.selectbox("Select student to register voice", list(st.session_state.students[class_name].keys()))
        student_data = st.session_state.students[class_name][selected_student]
        st.write("### 🎙️ Voice Registration")
        st.info("Record 5 voice samples. Student should say 'Present' or their name.")
        samples_recorded = len(student_data.get("voice_samples", []))
        st.progress(min(samples_recorded / 5, 1.0))
        st.write(f"**Samples recorded: {samples_recorded}/5**")
        if samples_recorded < 5:
            from streamlit_mic_recorder import mic_recorder
            audio = mic_recorder(start_prompt=f"🎤 Record Sample {samples_recorded + 1}", stop_prompt="⏹ Stop Recording", key=f"reg_{selected_student}_{samples_recorded}")
            if audio and audio.get("bytes"):
                audio_bytes = audio["bytes"]
                is_valid, message = validate_audio_quality(audio_bytes)
                st.write(message)
                if is_valid:
                    with st.spinner("🤖 Processing with AI..."):
                        text, confidence = transcribe_audio_whisper(audio_bytes=audio_bytes, language=whisper_language)
                        if text:
                            st.success(f"✅ Heard: '{text}' (Confidence: {confidence*100:.1f}%)")
                            embedding = create_voice_embedding(audio_bytes=audio_bytes)
                            if embedding is not None:
                                student_data["voice_samples"].append({
                                    "text": text,
                                    "embedding": embedding.tolist(),
                                    "confidence": float(confidence)
                                })
                                all_embeddings = [np.array(s["embedding"]) for s in student_data["voice_samples"]]
                                avg_embedding = np.mean(all_embeddings, axis=0)
                                student_data["voice_embedding"] = avg_embedding.tolist()
                                save_data()
                                if len(student_data["voice_samples"]) >= 5:
                                    student_data["registered"] = True
                                    save_data()
                                    st.balloons()
                                    st.success(f"🎉 {selected_student} successfully registered!")
        else:
            st.success("✅ Registration complete!")
        # Show photo + QR
        if student_data.get("photo"):
            img = Image.open(io.BytesIO(base64.b64decode(student_data["photo"])))
            st.image(img, width=120, caption=f"{selected_student}'s photo")
        qr_img = generate_qr_for_student({"name": selected_student, "roll_no": student_data["roll_no"]})
        st.markdown(get_image_download_link(qr_img, filename=f"{selected_student}_qr.png"), unsafe_allow_html=True)

# ----------------------------
# TAKE ATTENDANCE (roll-based with 5s timer) - REWRITTEN WITH WEBRTC
# ----------------------------
elif menu == "🎤 Take Attendance":
    st.subheader(f"🎤 AI Voice Attendance - {class_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Total Students", len(st.session_state.students[class_name]))
    with col2:
        today_count = len([a for a in st.session_state.attendance[class_name] if a.get("Date") == today_str])
        st.metric("✅ Present Today", today_count)
    with col3:
        absent = len(st.session_state.students[class_name]) - today_count
        st.metric("❌ Absent", absent)

    st.divider()
    st.info("Start class attendance. Click ▶ Start Attendance — then grant microphone permission. The system will take each student's attendance automatically.")

    # Start Attendance button (single click to start mic + flow)
    if st.button("▶ Start Attendance"):
        # prepare ordered list of students by roll_no if available
        st.session_state.attendance_active = True
        st.session_state.attendance_done = False
        st.session_state.recording_started = False
        st.session_state.countdown = COUNTDOWN_SECONDS_DEFAULT
        # Prepare ordered list (names only)
        ordered = sorted(list(st.session_state.students[class_name].items()), key=lambda x: (x[1].get("roll_no") or "", x[0]))
        st.session_state._attendance_order = [name for name, _ in ordered]
        st.session_state.current_roll_index = 0
        st.session_state.current_called = None
        st.session_state.processing_status = ""
        # Reset webrtc context if present
        st.session_state.webrtc_ctx = None
        st.rerun()

    if not st.session_state.attendance_active:
        st.info("Attendance not started. Click ▶ Start Attendance to begin.")
    else:
        attendees = st.session_state._attendance_order if st.session_state._attendance_order else []
        total_students = len(attendees)
        cur_idx = st.session_state.current_roll_index

        # -----------------------------------------------
        # 1. WebRTC Context Setup (Same, just ensuring it runs)
        # -----------------------------------------------
        webrtc_ctx = st.session_state.get("webrtc_ctx")
        if webrtc_ctx is None:
            # create a persistent audio processor that collects frames
            class AttendanceAudioProcessor(AudioProcessorBase):
                def __init__(self):
                    # Use a deque to ensure efficient appends/clears
                    self.frames = [] 
                    self.sample_rate = 48000  # assume 48k from browser
                    self.is_recording = False # Control when to collect frames

                def recv_audio(self, frame):
                    # Only collect frames if the flow explicitly sets is_recording to True
                    if self.is_recording:
                        arr = frame.to_ndarray()
                        # handle channel shape: flatten to 1-D mono by averaging channels if needed
                        if arr.ndim == 2:
                            # arr shape (n_channels, n_frames) — convert to (n_frames,)
                            arr_mono = np.mean(arr, axis=0)
                        else:
                            arr_mono = arr
                        self.frames.append(arr_mono)
                    return frame

            ctx = webrtc_streamer(
                key="attendance_webrtc",
                mode=WebRtcMode.SENDONLY, # Changed to SENDONLY, SENDRECV isn't strictly necessary for just mic
                media_stream_constraints={"audio": True, "video": False},
                audio_processor_factory=AttendanceAudioProcessor
            )

            # store context
            st.session_state.webrtc_ctx = ctx
            webrtc_ctx = ctx

        # -----------------------------------------------
        # 2. Flow Logic
        # -----------------------------------------------

        if not webrtc_ctx.state.playing:
            st.warning("🎤 Waiting for microphone permission. Please allow microphone access in the browser prompt.")
            # st.stop() # Avoid stopping if user is about to click "Allow"
        else:
            # If all processed
            if cur_idx >= total_students:
                st.success("🎉 The entire class attendance is complete.")
                st.session_state.attendance_active = False
                st.session_state.attendance_done = True
                
            else:
                current_student = attendees[cur_idx]
                current_student_info = st.session_state.students[class_name][current_student]

                # Display list with highlight
                st.write("### Class List")
                list_placeholder = st.empty()
                
                def update_list_display(current_idx, students_list, students_data):
                    list_markdown = ""
                    for idx, sname in enumerate(students_list):
                        display_roll = students_data[sname].get("roll_no", "")
                        if idx == current_idx:
                            list_markdown += f"<p style='color:red; font-weight:bold;'>➡ {display_roll} — {sname} (Your turn)</p>"
                        else:
                            list_markdown += f"<p>{display_roll} — {sname}</p>"
                    list_placeholder.markdown(list_markdown, unsafe_allow_html=True)
                
                update_list_display(cur_idx, attendees, st.session_state.students[class_name])


                # show current student info and status
                st.write("---")
                st.write(f"**Current:** Roll No. {current_student_info.get('roll_no','')} — **{current_student}**")
                status_placeholder = st.empty()
                status_placeholder.info("🔊 Calling student...")

                # --- 2A. CALL STUDENT (TTS PROMPT) ---
                if st.session_state.current_called != current_student:
                    # Speak the prompt only once per student
                    prompt_text = f"{current_student}, Roll number {current_student_info.get('roll_no','')}. Please say present."
                    speak_text(prompt_text, lang=whisper_language) 
                    st.session_state.current_called = current_student
                    # Give TTS time to start playing before starting countdown
                    time.sleep(1.5) 
                    st.rerun() # Rerun to start the recording phase

                # --- 2B. START RECORDING AND COUNTDOWN ---
                # Reset frames and start collection
                if webrtc_ctx.audio_processor:
                    webrtc_ctx.audio_processor.frames = []
                    webrtc_ctx.audio_processor.is_recording = True # Start collecting frames
                
                # Show countdown
                countdown_placeholder = st.empty()
                max_time = st.session_state.countdown # 5 seconds default
                start_time = time.time()
                
                # --- 2C. CONTINUOUS CHECK LOOP ---
                while (time.time() - start_time) < max_time:
                    seconds_left = max_time - int(time.time() - start_time)
                    countdown_placeholder.markdown(f"⏱ Time Left: **{seconds_left} sec**")
                    time.sleep(0.5) # Check every half second

                    # Process audio segment collected so far (e.g., last 1.5 seconds)
                    if webrtc_ctx.audio_processor and len(webrtc_ctx.audio_processor.frames) > 0:
                        
                        # Process ONLY the frames collected up to this point
                        frames_to_process = webrtc_ctx.audio_processor.frames
                        
                        # Concatenate and save to temp wav
                        try:
                            audio_np = np.concatenate(frames_to_process, axis=0)
                            temp_wav = f"temp_turn_{cur_idx}_{int(time.time())}.wav"
                            saved = save_np_audio_to_wav(audio_np, webrtc_ctx.audio_processor.sample_rate, temp_wav)
                        except Exception as e:
                            st.error(f"Error processing temp audio: {e}")
                            break # Break loop if fatal error

                        # Run validation & matching using the saved WAV (wav_path)
                        if saved:
                            
                            st.session_state.processing_status = "🤖 Checking voice..."
                            
                            text, text_confidence = transcribe_audio_whisper(wav_path=temp_wav, language=whisper_language)
                            emb = create_voice_embedding(wav_path=temp_wav)
                            
                            try:
                                os.remove(temp_wav) # Clean temp file immediately
                            except Exception:
                                pass
                            
                            if emb is not None:
                                matched, score, second = match_speaker(emb, class_name)
                                gap = score - (second or 0.0)
                                
                                # Decision logic: check if CURRENT student spoke
                                if matched == current_student and score >= speaker_threshold and gap >= speaker_gap and ("present" in text.lower() or "yes sir" in text.lower()):
                                    
                                    status_placeholder.success(f"✅ Voice matched: {current_student}! Score: {score:.3f}")
                                    
                                    # Mark present
                                    already_present = any(a["Name"]==matched and a["Date"]==today_str for a in st.session_state.attendance[class_name])
                                    if not already_present:
                                        st.session_state.attendance[class_name].append({
                                            "Name": matched,
                                            "Roll No": current_student_info["roll_no"],
                                            "Bus Student": current_student_info["bus"],
                                            "Local Student": current_student_info["local"],
                                            "Date": today_str,
                                            "Time": datetime.now().strftime("%H:%M:%S"),
                                            "Confidence": float(score),
                                            "Method": "AI Voice (Fast)"
                                        })
                                        save_data()
                                        speak_text(f"{matched} marked present. Thank you.")
                                    
                                    # Move to next student immediately
                                    webrtc_ctx.audio_processor.is_recording = False # Stop collecting
                                    st.session_state.current_roll_index += 1
                                    st.session_state.current_called = None # Reset called status
                                    time.sleep(1) # Give audio a moment to play
                                    st.rerun() # Immediately go to next student
                            
                            status_placeholder.markdown(f"**Heard:** '{text}' | Match: {matched} ({score:.3f}) | **Status:** {st.session_state.processing_status}")
                        
                        # Clear frames only if we didn't match the current student, to avoid re-processing the same audio segment.
                        # However, since we need to check if someone else is speaking, we keep the frames for the full time window (5s) for a final check.
                        
                # --- 2D. TIMEOUT / ABSENT LOGIC (After 5 seconds) ---
                
                # Stop recording after timeout
                if webrtc_ctx.audio_processor:
                    webrtc_ctx.audio_processor.is_recording = False 

                # If we reached here, the student was not marked present in the loop
                status_placeholder.error(f"⏱ {current_student} did not match or respond in time. Marked Absent.")
                
                st.session_state.attendance[class_name].append({
                    "Name": current_student,
                    "Roll No": current_student_info["roll_no"],
                    "Bus Student": current_student_info["bus"],
                    "Local Student": current_student_info["local"],
                    "Date": today_str,
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Confidence": 0.0,
                    "Method": "TimeoutAbsent"
                })
                save_data()
                
                # move to next
                st.session_state.current_roll_index += 1
                st.session_state.current_called = None
                time.sleep(0.5)
                st.rerun()

    # show today's attendance summary (same as before)
    st.divider()
    st.write("### ✅ Today's Attendance (so far)")
    today_attendance = [a for a in st.session_state.attendance[class_name] if a["Date"] == today_str]
    if today_attendance:
        for att in today_attendance:
            col1, col2, col3 = st.columns([2, 3, 1])
            col1.write(f"**{att['Roll No']}**")
            col2.write(f"{att['Name']}")
            col3.write(f"🎯 {att.get('Confidence', 0) * 100:.0f}%")
    else:
        st.info("No attendance marked yet today.")

    # Quick manual mark (fallback) (same as before)
    st.write("---")
    st.write("### Quick Manual Marking")
    students_list = list(st.session_state.students[class_name].keys())
    manual_name = st.selectbox("Select student to manually mark present", [""] + students_list)
    if st.button("Mark Present (Manual)"):
        if manual_name:
            info = st.session_state.students[class_name][manual_name]
            already_present = any(a["Name"] == manual_name and a["Date"] == today_str for a in st.session_state.attendance[class_name])
            if not already_present:
                st.session_state.attendance[class_name].append({
                    "Name": manual_name,
                    "Roll No": info["roll_no"],
                    "Bus Student": info["bus"],
                    "Local Student": info["local"],
                    "Date": today_str,
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Confidence": 1.0,
                    "Method": "Manual"
                })
                save_data()
                st.success(f"✅ {manual_name} marked present manually.")
            else:
                st.warning(f"{manual_name} already marked present.")

# ----------------------------
# VIEW REPORTS + Absent Report Generation (same as before)
# ----------------------------
elif menu == "📊 View Reports":
    st.subheader(f"📊 Attendance Reports - {class_name}")
    if len(st.session_state.attendance[class_name]) == 0:
        st.warning("No attendance records yet")
    else:
        for rec in st.session_state.attendance[class_name]:
            if "Date" not in rec:
                rec["Date"] = today_str
        dates = sorted(list(set([a["Date"] for a in st.session_state.attendance[class_name]])))
        selected_date = st.selectbox("📅 Select Date", dates, index=len(dates)-1)
        df = pd.DataFrame([a for a in st.session_state.attendance[class_name] if a["Date"]==selected_date])
        st.dataframe(df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            excel_path = f"attendance_{class_name}_{selected_date}.xlsx"
            df.to_excel(excel_path, index=False)
            with open(excel_path, "rb") as f:
                st.download_button("📘 Download Excel", f, file_name=excel_path)
        with col2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Attendance - {class_name} ({selected_date})", ln=True, align="C")
            pdf.ln(10)
            for _, row in df.iterrows():
                pdf.cell(200, 10, txt=f"{row['Roll No']} - {row['Name']} - {row['Time']}", ln=True)
            pdf_path = f"attendance_{class_name}_{selected_date}.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📕 Download PDF", f, file_name=pdf_path)
        st.divider()
        st.write("### 📝 Generate Absent Report")
        present_names = set(df["Name"].tolist())
        all_students = st.session_state.students[class_name]
        absentees = []
        for name, info in all_students.items():
            if name not in present_names:
                absentees.append({
                    "Name": name,
                    "Roll No": info["roll_no"],
                    "Bus Student": info["bus"],
                    "Local Student": info["local"]
                })
        if absentees:
            abs_df = pd.DataFrame(absentees)
            st.dataframe(abs_df, use_container_width=True)
            abs_excel = f"absent_{class_name}_{selected_date}.xlsx"
            abs_df.to_excel(abs_excel, index=False)
            with open(abs_excel, "rb") as f:
                st.download_button("📘 Download Absent Excel", f, file_name=abs_excel)
            pdf2 = FPDF()
            pdf2.add_page()
            pdf2.set_font("Arial", size=12)
            pdf2.cell(200, 10, txt=f"Absent Report - {class_name} ({selected_date})", ln=True, align="C")
            pdf2.ln(10)
            for a in absentees:
                pdf2.cell(200, 8, txt=f"{a['Roll No']} - {a['Name']} - Bus: {a['Bus Student']} - Local: {a['Local Student']}", ln=True)
            pdf2_path = f"absent_{class_name}_{selected_date}.pdf"
            pdf2.output(pdf2_path)
            with open(pdf2_path, "rb") as f:
                st.download_button("📕 Download Absent PDF", f, file_name=pdf2_path)
            txt = f"Absent Report for {class_name} on {selected_date}:\n" + "\n".join([f"{a['Roll No']} - {a['Name']}" for a in absentees])
            wa_url = "https://wa.me/?text=" + urllib.parse.quote(txt)
            st.markdown(f"[💬 Open WhatsApp Web to share absent report]({wa_url})", unsafe_allow_html=True)
        else:
            st.success("No absentees! All present.")

# ----------------------------
# Registered Students (same as before)
# ----------------------------
elif menu == "👥 Registered Students":
    st.subheader(f"👥 Registered Students - {class_name}")
    if len(st.session_state.students[class_name]) == 0:
        st.warning("No students registered yet")
    else:
        for name, info in st.session_state.students[class_name].items():
            with st.expander(f"{info['roll_no']} - {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Roll No:** {info['roll_no']}")
                    st.write(f"**Bus Student:** {info['bus']}")
                    st.write(f"**Local Student:** {info['local']}")
                with col2:
                    samples = len(info.get("voice_samples", []))
                    st.metric("Voice Samples", samples)
                    if info.get("registered"):
                        st.success("✅ Fully Registered")
                    else:
                        st.warning("⚠️ Incomplete")
                if info.get("photo"):
                    img = Image.open(io.BytesIO(base64.b64decode(info["photo"])))
                    st.image(img, width=120)
                qr_img = generate_qr_for_student({"name": name, "roll_no": info["roll_no"]})
                st.markdown(get_image_download_link(qr_img, filename=f"{name}_qr.png"), unsafe_allow_html=True)
                colA, colB = st.columns(2)
                if colA.button(f"Remove {name}", key=f"del_{name}"):
                    del st.session_state.students[class_name][name]
                    save_data()
                    st.success(f"Removed {name}")
                    st.rerun()
                if colB.button(f"Mark Present (Manual) - {name}", key=f"manual_{name}"):
                    already_present = any(a["Name"]==name and a["Date"]==today_str for a in st.session_state.attendance[class_name])
                    if not already_present:
                        st.session_state.attendance[class_name].append({
                            "Name": name,
                            "Roll No": info["roll_no"],
                            "Bus Student": info["bus"],
                            "Local Student": info["local"],
                            "Date": today_str,
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Confidence": 1.0,
                            "Method": "Manual"
                        })
                        save_data()
                        st.success(f"{name} marked present manually.")
                    else:
                        st.warning(f"{name} already marked present today")

# ----------------------------
# DASHBOARD (same as before)
# ----------------------------
elif menu == "📈 Dashboard":
    st.subheader(f"📈 Dashboard - {class_name}")
    att = st.session_state.attendance.get(class_name, [])
    if not att:
        st.warning("No attendance records yet")
    else:
        df = pd.DataFrame(att)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly = df.groupby('Month').size().reset_index(name='Count')
        st.write("#### Monthly Attendance")
        st.bar_chart(data=monthly.set_index('Month'))
        most_active = df.groupby('Name').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
        top = most_active.head(5)
        st.write("#### Most Active Students")
        st.table(top)
        bus_local = df.groupby('Bus Student').size().reset_index(name='Count')
        st.write("#### Bus vs Local (present records)")
        st.bar_chart(bus_local.set_index('Bus Student'))
        total_sessions = df['Date'].nunique()
        st.metric("Total recorded sessions", total_sessions)
        st.metric("Unique students who attended (records)", df['Name'].nunique())

# ----------------------------
# BACKUPS (same as before)
# ----------------------------
elif menu == "🔄 Backups":
    st.subheader("🔄 Backup Options")
    st.write("You can download the JSON files or backup to Firebase (if configured).")
    col1, col2 = st.columns(2)
    with col1:
        with open(STUDENTS_FILE, "rb") as f:
            st.download_button("📥 Download students_ai.json", f, file_name=STUDENTS_FILE)
    with col2:
        with open(ATTENDANCE_FILE, "rb") as f:
            st.download_button("📥 Download attendance_ai.json", f, file_name=ATTENDANCE_FILE)
    st.write("---")
    st.write("### 🔁 Firebase Backup (optional)")
    if not FIREBASE_AVAILABLE:
        st.info("Firebase admin SDK not installed. Install firebase-admin to enable.")
    else:
        sa = st.text_input("Path to Firebase Service Account JSON (leave blank to use env)", value=FIREBASE_SERVICE_ACCOUNT or "")
        if st.button("Backup to Firebase"):
            db = init_firebase(sa) if sa else init_firebase(FIREBASE_SERVICE_ACCOUNT)
            ok = backup_to_firebase(db)
            if ok:
                st.success("Backup to Firebase completed.")
            else:
                st.error("Firebase backup failed.")
    st.write("---")
    st.write("### Google Drive backup")
    st.info("Google Drive backup requires OAuth flow and is not auto-enabled here. If you want, I can add a Drive backup function (needs client_secret.json).")

# ----------------------------
# END
# ----------------------------
st.sidebar.write("Version: enhanced-v2.0")