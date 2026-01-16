import ollama

image_path1 = 'C:/Users/grabe_stud/RAG/RAG-Anything/docs/lama_pic.jpg'
image_path2 = 'C:/Users/grabe_stud/RAG/RAG-Anything/docs/sheep.jpg'

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'describe the animals shown in the images',
        # 'images': [image_path1, image_path2]
        'images': [image_path1]
    }],
    stream=False,
    # think='low'         # "llama3.2-vision" does not support thinking (status code: 400)
    format='',
    # format='json',
    options = {
        "temperature": 0.2,
        "num_ctx": 8192,
    }
)

print(response)
print(response["message"]["content"])