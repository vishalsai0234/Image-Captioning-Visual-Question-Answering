import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
import requests
from io import BytesIO
import streamlit as st


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Loaders (cached so they load only once) ────────────────────────────

@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_vqa_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(DEVICE)
    model.eval()
    return processor, model


# ─── Inference Functions ──────────────────────────────────────────────────────

def generate_caption(
    image: Image.Image,
    conditional_text: str = None,
    max_new_tokens: int = 50,
    num_beams: int = 5,
) -> str:
    processor, model = load_caption_model()

    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt").to(DEVICE)
    else:
        inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
    return processor.decode(output[0], skip_special_tokens=True)


def answer_question(
    image: Image.Image,
    question: str,
    max_new_tokens: int = 30,
) -> str:
    processor, model = load_vqa_model()
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")