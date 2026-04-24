# 🌍 Multilingual Hate Speech Detector

## 📌 Overview
This project is a Multilingual Hate Speech Detection System that analyzes text in multiple languages such as English, Arabic, and French.

The system uses a pretrained transformer model to classify the sentiment of text and then converts it into a final hate / not hate decision.

---

## 🎯 Project Objective
The main goal of this project is:
- Detect hate speech across multiple languages
- Use a single unified AI model instead of multiple models
- Demonstrate how AI can understand multilingual text

---

## 🧠 Model Used
The project uses a pretrained model from Hugging Face:

tabularisai/multilingual-sentiment-analysis

This model supports multiple languages and performs text classification.

---

## ⚙️ How It Works

Step 1: Input Text  
User provides text in any language (English, Arabic, French)

Step 2: Model Prediction  
The model returns sentiment:
- Positive
- Very Positive
- Negative
- Very Negative

Step 3: Decision Mapping  

Negative / Very Negative → Hate ❌  
Positive / Very Positive → Not Hate ✅  

---

## 🌍 Languages Supported
- English  
- Arabic  
- French  

---

## 📁 Project Structure

multilingual-hate-speech-detector/
│
├── data/
│   └── dataset_link.txt
│
├── models/
│
├── results/
│   ├── results.txt
│   └── output.png
│
├── src/
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── main.py  
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## ▶️ How to Run

1. Install requirements:
pip install -r requirements.txt

2. Run the project:
python main.py

---

## 📊 Sample Results

Text: I hate you  
Model Output: Negative  
Final Decision: Hate ❌  

Text: You are amazing  
Model Output: Very Positive  
Final Decision: Not Hate ✅  

Text: أكرهك  
Model Output: Negative  
Final Decision: Hate ❌  

Text: أنت شخص رائع  
Model Output: Positive  
Final Decision: Not Hate ✅  

Text: Je te déteste  
Model Output: Very Negative  
Final Decision: Hate ❌  

---

## 📸 Output Screenshot

![Output Screenshot](results/output.png)

---

## 🛠️ Technologies Used
- Python  
- Hugging Face Transformers  
- PyTorch  
- Multilingual NLP Model  

---

## 📚 Dataset Reference
Multilingual HatEval 2019 Dataset:
https://github.com/msang/hateval

---

## ✅ Conclusion
This project demonstrates a simple multilingual hate speech detection system using a pretrained transformer model.

The system can analyze multiple languages and detect harmful content effectively.

---

## 🚀 Future Work
- Fine-tune model on real hate speech dataset  
- Improve Arabic detection  
- Add evaluation metrics  
- Build web interface  

---

## 👨‍💻 Author
Karrar Haider