import google.generativeai as genai
genai.configure(api_key="insert_your_key_here")

models = genai.list_models()

for model in models:
    print(f"Name: {model.name} | Supports generation: {'generateContent' in model.supported_generation_methods}")
