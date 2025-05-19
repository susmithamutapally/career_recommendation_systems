# app.py

import streamlit as st
from openai import OpenAI
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# === Setup ===
st.set_page_config(page_title="Career Recommender", layout="centered")
st.title("🎓 Career Path Recommender")
st.write("Answer a few questions to get a personalized career suggestion!")

# === Inputs ===
interests = st.multiselect(
    "What are your interests?",
    ["Technology", "Business", "Healthcare", "Engineering", "Media", "Agriculture", "Finance", "AI", "Design", "Education"]
)

skills = st.multiselect(
    "What skills do you have?",
    ["Problem-solving", "Coding", "Math", "Writing", "Creativity", "Teamwork", "Public speaking"]
)

education = st.selectbox(
    "What is your current education level?",
    ["Grade 10", "Grade 11", "Grade 12", "Post-matric / University"]
)

submit = st.button("Find My Career Path")

# === Load Karrierewege model ===
@st.cache_resource
def load_model():
    model_name = "ElenaSenger/career-path-representation-mpnet-karrierewege"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# === If user submits ===
if submit:
    # === Step 1: Generate starting job using ChatGPT ===
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    # openai.api_key="anything"
    # openai.base_url="http://localhost:3040/v1"
    prompt = f"""
    Based on the following student profile, suggest one starting job they could pursue after school (e.g., IT Assistant, Lab Technician, Sales Trainee). Be concise.

    Interests: {', '.join(interests)}
    Skills: {', '.join(skills)}
    Education: {education}

    Only return the job title and nothing else.
    """

    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful career advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            starting_job = response.choices[0].message.content.strip()
            st.success(f"🧭 Starting Role: {starting_job}")

            # === Step 2: Karrierewege inference ===
            candidate_jobs = [
                "Software Developer", "Marketing Specialist", "Data Analyst",
                "AI Researcher", "Mechanical Engineer", "Healthcare Assistant",
                "Teacher", "Sales Manager", "UX Designer", "Lab Technician"
            ]

            # Embed history
            history_inputs = tokenizer(starting_job, return_tensors="pt")
            with torch.no_grad():
                history_output = model(**history_inputs)
                history_embedding = history_output.pooler_output

            # Embed candidates
            candidate_inputs = tokenizer(candidate_jobs, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                candidate_output = model(**candidate_inputs)
                candidate_embeddings = candidate_output.pooler_output

            # Cosine similarity
            similarities = F.cosine_similarity(history_embedding, candidate_embeddings)
            top_scores, top_indices = similarities.topk(3)

            st.write("🔮 **Top Recommended Career Paths:**")
            for i in top_indices:
                st.write(f"• {candidate_jobs[i]} (Score: {similarities[i]:.2f})")

    except Exception as e:
        st.error(f"Error generating career suggestion: {e}")
