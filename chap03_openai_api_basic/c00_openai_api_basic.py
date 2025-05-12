from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": "농촌진흥청 농촌인적자원개발센터에 대해 설명해줘"
        }
    ]
)

print(completion.choices[0].message.content)