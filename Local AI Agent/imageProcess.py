import base64
import os
from typing import Optional, cast
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

load_dotenv()

class Classification(BaseModel):
    is_receipt: bool


class Item(BaseModel):
    name: str
    price: Optional[float] = None


class Receipt(BaseModel):
    org_name: Optional[str] = None
    date: Optional[str] = None
    total: Optional[float] = None
    tax: Optional[float] = None
    items: list[Item] = []


_llm = ChatOllama(model='minicpm-v', base_url='http://localhost:11434', temperature=0.1)
classifier = _llm.with_structured_output(Classification)
receipt_extractor = _llm.with_structured_output(Receipt)

CLASSIFY_PROMPT = "Is this image a receipt or invoice? Answer true or false."

EXTRACT_PROMPT = """Extract the receipt information from this image:
- org_name: name of the store or restaurant
- date: date on the receipt (YYYY-MM-DD if possible)
- total: total amount as a number
- tax: tax amount as a number
- items: list of items, each with name and price as a number
If a field is not visible, leave it as null."""

SUMMARY_PROMPT = "Describe the contents of this image in a few sentences."

IMAGES_FOLDER = 'images'

for filename in os.listdir(IMAGES_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        continue

    filepath = os.path.join(IMAGES_FOLDER, filename)
    with open(filepath, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    # [-1] = last element
    ext = filename.rsplit('.', 1)[-1].lower()
    # mime => type/subtype, e.g. image/jpeg or image/png
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'

    # Base64 converts binary → text
    def make_message(prompt):
        # Multi-modal input (text + image)
        return HumanMessage(content=[
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_base64}'}}
        ])

    print(f'--- {filename} ---')

    result = cast(Classification, classifier.invoke([make_message(CLASSIFY_PROMPT)]))

    if result.is_receipt:
        print('[Receipt detected]')
        receipt = cast(Receipt, receipt_extractor.invoke([make_message(EXTRACT_PROMPT)]))
        print(receipt.model_dump_json(indent=2))
    else:
        print('[Not a receipt]')
        summary = _llm.invoke([make_message(SUMMARY_PROMPT)])
        print(summary.content)

    print()
