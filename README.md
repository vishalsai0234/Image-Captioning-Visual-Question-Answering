# DA627 – Image Captioning & Visual Question Answering

A multimodal AI project for the course **DA627: Building Multimodal GenAI**.

This project implements two core vision–language tasks:

* **Image Captioning** – generating natural language descriptions from images
* **Visual Question Answering (VQA)** – answering questions about image content

The system is built using pretrained vision–language models from Hugging Face and provides an interactive web interface using Streamlit.

<img width="1920" height="931" alt="02" src="https://github.com/user-attachments/assets/22a80458-bfb4-4a79-96d7-308f5bb03f4b" />

---

# Features

* Generate captions for uploaded images
* Ask natural language questions about an image
* Evaluate model outputs using standard metrics
* Interactive Streamlit interface with multiple pages
* Supports both uploaded images and sample images
* Uses pretrained BLIP models for captioning and VQA

---

# Models Used

| Model                                   | Task                      | Source       |
| --------------------------------------- | ------------------------- | ------------ |
| `Salesforce/blip-image-captioning-base` | Image Captioning          | Hugging Face |
| `Salesforce/blip-vqa-base`              | Visual Question Answering | Hugging Face |

---

# Project Structure

```text
DA627-VQA-Captioning/
│
├── app.py
├── evaluation.py
├── models.py
├── requirements.txt
├── Image Captioning & VQA.ipynb # Jupyter Notebook
│
├── sample_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   ├── image5.jpg
│   └── image6.jpg
│
├── datasets/
│   ├── Flickr8k/
│   └── VQA/
│
└── streamlit_results/
    ├── Home page.png
    ├── Image caption generation.png
    ├── Visual question answering.png
    ├── Evaluation - Caption Metrics.png
    ├── Evaluation - VQA Accuracy.png
    └── Project Overview.png
```

---

# Software Architecture

## `models.py`

Contains:

* Model loading with `@st.cache_resource`
* Image loading and preprocessing utilities
* `generate_caption(image)`
* `answer_question(image, question)`

## `evaluation.py`

Contains from-scratch implementations of:

* BLEU
* METEOR
* CIDEr
* VQA Accuracy

## `app.py`

Implements a five-page Streamlit application:

1. Home
2. Image Captioning
3. Visual Q&A
4. Evaluation
5. About

---

# Datasets

## Flickr8k Dataset

Used for image captioning.

* 8,000 images
* 5 captions per image
* Suitable for benchmarking caption generation

Dataset: [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## VQA v2 Dataset

Used for visual question answering.

* Image-question-answer triplets
* Includes open-ended and yes/no questions

Dataset: [https://visualqa.org/download.html](https://visualqa.org/download.html)

---

# Evaluation Metrics

## Image Captioning

* **BLEU-1 to BLEU-4** – n-gram overlap with reference captions
* **METEOR** – synonym-aware alignment metric
* **CIDEr** – TF-IDF weighted caption similarity

## Visual Question Answering

* **Accuracy** – exact match between predicted and reference answers

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DA627-VQA-Captioning.git
cd DA627-VQA-Captioning
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# Example Usage

## Image Captioning

1. Open the Image Captioning page
2. Upload an image or select a sample image
3. Click Generate Caption
4. The model produces a caption

Example:

```text
Input Image: Dog playing in a park
Generated Caption: "A dog running through the grass"
```

## Visual Question Answering

1. Open the Visual Q&A page
2. Upload an image
3. Enter a question such as:

```text
What is the person holding?
```

4. Click Answer Question
5. The model returns an answer

---

# Future Improvements

* Add support for more advanced models such as BLIP-2 or LLaVA
* Compare BLIP with CLIP-based retrieval approaches
* Add support for batch evaluation on datasets
* Deploy the app online using Streamlit Cloud or Hugging Face Spaces
* Add attention visualizations to explain model predictions

---

# References

1. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*.
2. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*.
3. Li, J., Li, D., Savarese, S., & Hoi, S. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*.
4. Antol, S. et al. (2015). *VQA: Visual Question Answering*.

---

# Author

**Vishal**
Roll No.: 220150029

DA627 – Building Multimodal GenAI

Indian Institute of Technology Guwahati
