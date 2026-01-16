import ollama

print("=======================TEXT-only============================")
print("=============================================================")

# text only
prompt = "Who/which company built the llm model llama3.2-vision:11b-instruct-q4_K_M"
response1 = ollama.chat(
    model='llama3.2-vision:11b-instruct-q4_K_M',
    messages=[{
        "role": "user", "content": prompt
    }]
)
result1 = response1["message"]["content"]
print(result1)

print("=======================Image-only============================")
print("=============================================================")

# image 
image_path = 'C:/Users/grabe_stud/RAG/RAG-Anything/docs/lama_pic.jpg'
response2 = ollama.chat(
    model='llama3.2-vision:11b-instruct-q4_K_M',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': [image_path]
    }]
)
result2 = response2["message"]["content"]
print(result2)

print("=======================Multimodal============================")
print("=============================================================")

# multimodal
system_prompt = "You are a LLM expert."
# image_mode = True
prompt = "Describe the picture in 2 sentences"
image_path = 'C:/Users/grabe_stud/RAG/RAG-Anything/docs/lama_pic.jpg'


# Read the image and encode it in Base64
import base64
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

if system_prompt:
    system_message = {"role": "system", "content": system_prompt}
else:
    system_message = None

if image_data:
    user_message = {
        "role": "user",
        "content": prompt,
        "images": [image_path],
    }
else:
    user_message = {"role": "user", "content": prompt}

messages_multimodal = [system_message, user_message]

response3 = ollama.chat(
    model='llama3.2-vision:11b-instruct-q4_K_M',
    messages=messages_multimodal
)
result3 = response3["message"]["content"]
print(result3)