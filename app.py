import streamlit as st
import threading
import tempfile
import time
from eye_detector import EyeTracker
from audiologic import AudioLOGIC
from interviewlogic import InterviewLOGIC

# Initialize classes
audio_logic = AudioLOGIC()
interview_logic = InterviewLOGIC()

# Streamlit app layout
st.set_page_config(page_title="SpeaKoach", layout="wide")
st.title("SpeaKoach")

# Upload resume
uploaded_pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
company = st.text_input("Company name:")
position = st.text_input("Position applied for:")
num_questions = st.number_input("Number of questions", 1, 5, 3)
speak_duration = st.number_input("Answer duration (seconds)", 5, 60, 15)

# Create placeholders for dynamic updates
status_text = st.empty()
gaze_score_text = st.empty()
question_box = st.empty()
answer_box = st.empty()

if uploaded_pdf and st.button("🚀 Start Interview"):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_pdf.read())
        pdf_path = tmp_pdf.name

    tracker = EyeTracker()
    t1 = threading.Thread(target=tracker.start_tracking)
    t1.start()

    st.success("Interview started! Please look at the camera.")
    time.sleep(2)

    resume_text = interview_logic.extract_text_from_pdf(pdf_path)
    QA_pairs = []

    for i in range(num_questions):
        question = interview_logic.generate_question(resume_text, company, position, QA_pairs)
        question_box.markdown(f"**Q{i + 1}: {question}**")
        audio_logic.speak_text(question)
        status_text.text("🎙️ Recording your answer...")


        filename = audio_logic.record_audio(duration=speak_duration)
        answer = audio_logic.transcribe_audio(filename)
        answer_box.markdown(f"**You:** {answer}")
        QA_pairs.append((question, answer))

        if any(word in answer.lower() for word in ["stop now", "quit", "exit"]):
            status_text.text("Interview stopped early.")
            break


        gaze_score_text.text(f"👁️ Eye contact so far: {tracker.center_count} center gaze detections")

    # Stop tracking
    tracker.running = False
    t1.join()

    # Generate conclusion
    st.subheader("📋 Interview Summary")
    conclusion = interview_logic.generate_conclusion(resume_text, QA_pairs)
    st.write(conclusion)

    st.metric("Average Eye Contact (%)", f"{tracker.gaze_score_ratio:.1f}")

    # Optional: audio feedback
    audio_logic.speak_text(conclusion)


