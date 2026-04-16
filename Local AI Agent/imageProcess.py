import base64
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOllama(model='bakllava', base_url='http://localhost:11434')

IMAGES_FOLDER = 'images'

for filename in os.listdir(IMAGES_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        continue

    filepath = os.path.join(IMAGES_FOLDER, filename)
    with open(filepath, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    ext = filename.rsplit('.', 1)[-1].lower()
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'

    message = HumanMessage(content=[
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_base64}'}}
    ])

    print(f'--- {filename} ---')
    response = model.invoke([message])
    print(response.content)
    print()
