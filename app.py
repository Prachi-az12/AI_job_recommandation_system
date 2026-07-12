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

tfidf = pickle.load(
    open("tfidf.pkl", "rb")
)

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

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text



# ---------------- Resume Validation ----------------

def check_resume(text):

    import re

    text = text.lower()

    # Email check
    has_email = bool(
        re.search(r'\S+@\S+\.\S+', text)
    )

    # Phone check
    has_phone = bool(
        re.search(r'\b\d{10}\b', text)
    )


    # Mandatory Resume Sections

    has_skills = (
        "skills" in text or
        "technical skills" in text
    )

    has_education = (
        "education" in text or
        "qualification" in text
    )

    has_experience = (
        "experience" in text or
        "internship" in text or
        "project" in text
    )


    # Resume Validation

    if (
        has_skills
        and has_education
        and has_experience
        and (has_email or has_phone)
    ):
        return True

    else:
        return False


# ---------------- Skill Extraction ----------------


skill_list = [

    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "html",
    "css",
    "javascript",
    "react",
    
    
    "flask",
    "machine learning",
    "deep learning",
    "nlp",
    "pandas",
    "numpy",
    "data analysis",
    "data science",
    "excel",
   

]


def extract_skills(text):

    text = text.lower()

    skills = []

    for skill in skill_list:

        if skill in text:
            skills.append(skill)

    return list(set(skills))



# ---------------- Job Recommendation ----------------


def recommend_job(resume_text):

    resume_vector = tfidf.transform(
        [resume_text]
    )


    similarity = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]


    top_index = similarity.argsort()[-5:][::-1]


    result = jobs.iloc[top_index].copy()


    result["Match Score"] = (
        similarity[top_index] * 100
    ).round(2)


    return result



# ---------------- Recommendation ----------------

# ---------------- Recommendation ----------------
# if st.button("🚀 Find Suitable Jobs"):
#     result = recommend_job(resume_text)
#     st.subheader("🎯 Recommended Jobs")
#     for index, row in result.iterrows():
#         st.markdown(
#             f"""
#             <div class='job-card'>
#                 <div class='job-title'>💼 {row['job_title']}</div>
#                 <br>
#                 <b>🎯 Match Score:</b> {row['Match Score']}%
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     # Download Report
#     csv = result[['job_title','Match Score']].to_csv(index=False)
#     st.download_button(
#         label="📥 Download Recommendation Report",
#         data=csv,
#         file_name="Job_Recommendation_Report.csv",
#         mime="text/csv"
#     )

# ---------------- Custom CSS ----------------

st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
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
.skill-box{
    background:#dbeafe;
    padding:8px 15px;
    border-radius:20px;
    margin:5px;
    display:inline-block;
    color:#1e40af;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# इमेज अॅड करा (width वापरून)
st.image(
    "https://copilot.microsoft.com/th/id/BCO.ca96c099-c07c-4c79-9d8b-5fa0bad02520.png",
    #"https://copilot.microsoft.com/th/id/BCO.7a3f9b2d-4e2a-4f9c-9c3a-8f2c7e6a9a11.png",
    #"https://copilot.microsoft.com/th/id/BCO.9c5d8f44-2b6e-4a7e-8e9d-1a2b3c4d5e67.png",
    width=500  
)





# ---------------- Sidebar ----------------

st.sidebar.title("💼 Project Information")

st.sidebar.write("""

### AI Job Recommendation System


Technology Used:

🔹 NLP

🔹 TF-IDF Vectorization

🔹 Cosine Similarity

🔹 Machine Learning


Features:

✅ Resume Analysis

✅ Skill Extraction

✅ Job Recommendation

✅ Download Report

""")


st.sidebar.info(
    f"Total Jobs Available : {len(jobs)}"
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



# ---------------- Upload Resume ----------------


uploaded_file = st.file_uploader(
    "📄 Upload Your Resume PDF",
    type=["pdf"]
)



if uploaded_file:


    resume_text = extract_resume_text(
        uploaded_file
    )


    # Empty PDF Check

    if resume_text.strip() == "":

        st.error(
            "❌ No text found in PDF."
        )

        st.stop()



    # Resume Validation

    if not check_resume(resume_text):

        st.error(
            "❌ Please upload a valid Resume PDF only."
        )

        st.info(
            "This document does not look like a resume."
        )

        st.stop()



    st.success(
        "✅ Resume Uploaded Successfully"
    )



    # Resume Preview

    with st.expander("📄 Resume Preview"):

        st.write(
            resume_text[:1500]
        )



    # Skills


    skills = extract_skills(
        resume_text
    )


    st.subheader(
        "🧠 Detected Skills"
    )


    if skills:


        for skill in skills:

            st.markdown(

                f"""
                <span class='skill-box'>
                {skill}
                </span>
                """,

                unsafe_allow_html=True
            )


    else:

        st.warning(
            "⚠️ No skills detected."
        )



    # Recommendation


        # Recommendation
    if st.button("🚀 Find Suitable Jobs"):
        result = recommend_job(resume_text)
        st.subheader("🎯 Recommended Jobs")
        for index, row in result.iterrows():
            st.markdown(
                f"""
                <div class='job-card'>
                    <div class='job-title'>💼 {row['job_title']}</div>
                    <br>
                    <b>🎯 Match Score:</b> {row['Match Score']}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # Download Report
        csv = result[['job_title','Match Score']].to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendation Report",
            data=csv,
            file_name="Job_Recommendation_Report.csv",
            mime="text/csv"
        )
