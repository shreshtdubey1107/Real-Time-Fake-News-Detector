
from ollama import chat

def verify_with_phi(article):

    prompt = f"""
You are an expert fake news detector.

Estimate the probability that the article is TRUE.

Respond with ONLY ONE INTEGER between 0 and 100.

Examples:

98

72

45

Do not explain.
Do not write any words.
Do not write any punctuation.


Article:

{article}
"""

    response = chat(
        model="phi3:mini",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    text = response["message"]["content"].strip()


    

    probability = float(text)
    

    return probability 