def generate_prompt(user_interests: str, skills: str, goals: str) -> str:
    prompt = (
        f"Based on the following:\n"
        f"- Interests: {user_interests}\n"
        f"- Skills: {skills}\n"
        f"- Goals: {goals}\n\n"
        f"Suggest 3 suitable career paths and explain the reasoning for each."
    )
    return prompt
