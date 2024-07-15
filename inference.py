import base64
import json
import os
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from peft import PeftModel, PeftConfig
from torchvision import transforms
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

app = FastAPI()

# Load the model and configuration
model_name = "HuggingFaceM4/idefics2-8b"
checkpoint_path = "checkpoint-5000"

print("Loading the model...")
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)
# model = Idefics2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
config = PeftConfig.from_pretrained(checkpoint_path)
# model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
# model.print_trainable_parameters()


# Define the request body schema
class PredictionRequest(BaseModel):
    OCR: str
    image: str  # Assume this is base64 encoded image string
    entities: Dict[str, str]

class PredictionResponse(BaseModel):
    output_text: str


def process_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((980, 980)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def tensorise_input(request: PredictionRequest):
    def gen_question(ocr_text, entities):
        res_str = f"Given the meme and its text: " + f'"{ocr_text}"' + f" , group the entities: {entities} in regards to the predefined format."
        return res_str

    image = process_base64_image(request.image)
    ocr_text = request.OCR
    entities = request.entities
    question = gen_question(ocr_text, entities)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer in the following format: {hero: [], villain: [], victim: []}"},
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)

    return inputs

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    def format_pred(pred_str):
        pred_str = pred_str.replace("{hero: [", '{"hero": [')
        pred_str = pred_str.replace(" villain: [", ' "villain": [')
        pred_str = pred_str.replace(" victim: [", ' "victim": [')

        pred_str = pred_str.replace("'", '"')

        return json.loads(pred_str)

    try:
        # Process input and generate predictions
        inputs = tensorise_input(request)
        # outputs = model.generate(**inputs, max_new_tokens=64)
        # output_text = processor.batch_decode(outputs[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        # output_text = format_pred(output_text[0])
        output_text = "test"

        return PredictionResponse(output_text=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
