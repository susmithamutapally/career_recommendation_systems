# 🎓 Career Path Recommender

A personalized, AI-powered career recommendation tool developed using OpenAI and Hugging Face models. This interactive Streamlit app helps students and early professionals discover suitable career paths based on their interests, skills, and education level.

<!-- ![App Screenshot](https://your-screenshot-url-if-hosted.com) Optional: Add a screenshot link -->

---

<!-- ## 🚀 Live Demo -->

<!-- 👉 [Try the App](https://your-streamlit-app-url.streamlit.app) -->

---

## How It Works

1. **User Inputs**: Choose your interests, skills, and current education level.
2. **Role Generation**: An OpenAI model suggests a suitable starting role.
3. **Career Path Matching**: A pre-trained Hugging Face model computes embeddings for the starting role and compares it with a set of potential career options using cosine similarity.
4. **Top Career Matches**: The app recommends the most aligned next steps based on semantic similarity.

---

## Example

**Inputs:**
- Interests: Business
- Skills: Problem-solving
- Education Level: Post-matric / University

**Output:**
- Starting Role: *Business Analyst*
- Top Career Matches:
  - Data Analyst
  - Marketing Specialist
  - Product Manager

---

## Tech Stack

- [Streamlit](https://streamlit.io) — frontend UI
- [OpenAI API](https://platform.openai.com) — for starting role generation
- [Hugging Face Transformers](https://huggingface.co) — for semantic similarity (`mpnet-karrierewege`)
- [PyTorch](https://pytorch.org) — model inference
- [scikit-learn](https://scikit-learn.org) — similarity computations

---

