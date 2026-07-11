import streamlit as st
import pickle
import pdfplumber
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Load Model Files ----------------

import gzip

tfidf = pickle.load(open("tfidf.pkl", "rb"))

job_vectors = pickle.load(
    gzip.open("job_vectors.pkl.gz", "rb")
)

jobs = pickle.load(
    gzip.open("jobs.pkl.gz", "rb")
)

# ---------------- Resume Text Extraction ----------------

def extract_resume_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text



# ---------------- Skill Extraction ----------------

skill_list = [
    "python",
    "sql",
    "machine learning",
    "nlp",
    "pandas",
    "numpy",
    "data analysis",
    "java",
    "html",
    "css",
    "javascript"
]


def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skill_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills



# ---------------- Job Recommendation ----------------

def recommend_job(resume_text):

    resume_vector = tfidf.transform([resume_text])


    similarity_scores = cosine_similarity(
        resume_vector,
        job_vectors
    )


    top_indices = similarity_scores[0].argsort()[-5:][::-1]


    recommended_jobs = jobs.iloc[top_indices]


    return recommended_jobs[
        ["job_title", "job_description"]
    ]



# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)



# ---------------- Custom CSS ----------------

st.markdown("""
<style>

.stApp{

background: linear-gradient(
135deg,
#eef2ff,
#f8fafc
);

}


.main-title{

font-size:45px;
font-weight:bold;
text-align:center;
color:#1e3a8a;

}


.sub-title{

text-align:center;
font-size:18px;
color:#475569;
margin-bottom:30px;

}



.job-card{

background:white;
padding:20px;
border-radius:15px;
margin:15px 0px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);

}



.job-title{

font-size:22px;
font-weight:bold;
color:#2563eb;

}


.job-desc{

font-size:15px;
color:#334155;

}


</style>

""", unsafe_allow_html=True)



# ---------------- Sidebar ----------------

st.sidebar.title("💼 Project Info")


st.sidebar.write(
"""
**AI Job Recommendation System**

Built Using:

🔹 NLP  
🔹 TF-IDF Vectorization  
🔹 Cosine Similarity  

Features:

✅ Resume Analysis  
✅ Skill Extraction  
✅ Job Recommendation  

"""
)


st.sidebar.info(
f"Total Jobs Available: {len(jobs)}"
)



# ---------------- Main UI ----------------


st.markdown(
"<div class='main-title'>💼 AI Job Recommendation System</div>",
unsafe_allow_html=True
)


st.markdown(
"<div class='sub-title'>NLP Based Resume Analysis & Job Matching System</div>",
unsafe_allow_html=True
)



uploaded_file = st.file_uploader(
"📄 Upload Your Resume PDF",
type=["pdf"]
)



if uploaded_file:


    resume_text = extract_resume_text(uploaded_file)


    st.success(
        "Resume Uploaded Successfully ✅"
    )



    # Skills Display

    skills = extract_skills(resume_text)


    st.subheader("🧠 Detected Skills")


    if skills:

        for skill in skills:

            st.badge(skill)

    else:

        st.write("No skills detected")



    # Recommendation Button


    if st.button("🚀 Find Suitable Jobs"):


        result = recommend_job(resume_text)


        st.subheader(
            "🎯 Recommended Jobs"
        )



        for index,row in result.iterrows():


            st.markdown(

            f"""

            <div class='job-card'>


            <div class='job-title'>

            💼 {row['job_title']}

            </div>


            <br>


            <div class='job-desc'>

            {row['job_description'][:350]}...

            </div>


            </div>

            """,

            unsafe_allow_html=True

            )



        # Download Report


        csv = result.to_csv(index=False)


        st.download_button(

            label="📥 Download Recommendation Report",

            data=csv,

            file_name="Job_Recommendation_Report.csv",

            mime="text/csv"

        )
