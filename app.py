import streamlit as st
import pickle
import gzip
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# ---------------- Load Model Files ----------------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
job_vectors = pickle.load(gzip.open("job_vectors.pkl.gz", "rb"))
jobs = pickle.load(gzip.open("jobs.pkl.gz", "rb"))

# ---------------- Resume Text Extraction ----------------
def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

# ---------------- Resume Validation ----------------
def check_resume(text):
    import re
    text = text.lower()
    has_email = bool(re.search(r'\S+@\S+\.\S+', text))
    has_phone = bool(re.search(r'\b\d{10}\b', text))
    has_skills = ("skills" in text or "technical skills" in text)
    has_education = ("education" in text or "qualification" in text)
    has_experience = ("experience" in text or "internship" in text or "project" in text)
    return has_skills and has_education and has_experience and (has_email or has_phone)

# ---------------- Skill Extraction ----------------
skill_list = ["python","java","c","c++","sql","mysql","html","css","javascript","react",
              "flask","machine learning","deep learning","nlp","pandas","numpy",
              "data analysis","data science","excel"]

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in skill_list if skill in text]))

# ---------------- Job Recommendation ----------------
def recommend_job(resume_text):
    resume_vector = tfidf.transform([resume_text])
    similarity = cosine_similarity(resume_vector, job_vectors)[0]
    top_index = similarity.argsort()[-5:][::-1]
    result = jobs.iloc[top_index].copy()
    result["Match Score"] = (similarity[top_index] * 100).round(2)
    return result

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#e3f2fd,#bbdefb);
}
.main-title {
    font-size:45px;
    font-weight:bold;
    text-align:center;
    background: linear-gradient(to right,#1565c0,#1e88e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align:center;
    font-size:18px;
    color:#37474f;
    margin-bottom:30px;
    font-style:italic;
}
.job-card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    padding:20px;
    border-radius:15px;
    margin:15px 0px;
    box-shadow:0px 6px 15px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.job-card:hover {
    transform: scale(1.02);
    box-shadow:0px 10px 20px rgba(0,0,0,0.25);
}
.job-title {
    font-size:22px;
    font-weight:bold;
    color:#0d47a1;
    margin-bottom:10px;
}
.skill-box {
    background: linear-gradient(to right,#90caf9,#64b5f6);
    padding:8px 15px;
    border-radius:20px;
    margin:5px;
    display:inline-block;
    color:#0d47a1;
    font-weight:bold;
    box-shadow:0px 2px 6px rgba(0,0,0,0.1);
    transition: all 0.2s ease;
}
.skill-box:hover {
    background: linear-gradient(to right,#1e88e5,#1565c0);
    color:white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #bbdefb, #90caf9);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}
img {
    border-radius: 15px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=100)
st.sidebar.markdown("<div class='sidebar-title'>💼 Job Recommendation System</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>✨ Upload your resume and get matched instantly!</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-highlight'>🔍 Powered by NLP + TF-IDF + Cosine Similarity</div>", unsafe_allow_html=True)
st.sidebar.write("""
### ⚡ Technology Used
- NLP
- TF-IDF Vectorization
- Cosine Similarity
- Machine Learning

### 🎯 Features
- Resume Analysis
- Skill Extraction
- Job Recommendation
- Download Report
""")
st.sidebar.info(f"Total Jobs Available : {len(jobs)}")

# ---------------- Main UI ----------------
st.markdown("<div class='main-title'>💼 AI Job Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>NLP Based Resume Analysis & Job Matching System</div>", unsafe_allow_html=True)

# Blue themed image
st.image(
    "https://copilot.microsoft.com/th/id/BCO.ca96c099-c07c-4c79-9d8b-5fa0bad02520.png",
    width=500
)


# ---------------- Upload Resume ----------------
uploaded_file = st.file_uploader("📄 Upload Your Resume PDF", type=["pdf"])

if uploaded_file:
    resume_text = extract_resume_text(uploaded_file)
    if resume_text.strip() == "":
        st.error("❌ No text found in PDF.")
        st.stop()
    if not check_resume(resume_text):
        st.error("❌ Please upload a valid Resume PDF only.")
        st.info("This document does not look like a resume.")
        st.stop()
    st.success("✅ Resume Uploaded Successfully")

    with st.expander("📄 Resume Preview"):
        st.write(resume_text[:1500])

    skills = extract_skills(resume_text)
    st.subheader("🧠 Detected Skills")
    if skills:
        for skill in skills:
            st.markdown(f"<span class='skill-box'>{skill}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No skills detected.")

    if st.button("🚀 Find Suitable Jobs"):
        result = recommend_job(resume_text)
        st.subheader("🎯 Recommended Jobs")
        for _, row in result.iterrows():
            st.markdown(f"""
            <div class='job-card'>
                <div class='job-title'>💼 {row['job_title']}</div>
                <b>🎯 Match Score:</b> {row['Match Score']}%
            </div>
            """, unsafe_allow_html=True)
        csv = result[['job_title','Match Score']].to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendation Report",
            data=csv,
            file_name="Job_Recommendation_Report.csv",
            mime="text/csv"
        )





