from transformers import pipeline

print("Loading model... please wait...\n")

classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis"
)

print("=== Multilingual Hate Speech Detector ===")
print("=" * 50)

examples = [
    "I hate you",
    "You are amazing",
    "أكرهك",
    "أنت شخص رائع",
    "Je te déteste",
    "Tu es gentil"
]

for text in examples:
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # تحويل النتيجة
    if label.lower() == "negative":
        final_label = "Hate ❌"
    else:
        final_label = "Not Hate ✅"

    print(f"\nText: {text}")
    print(f"Model Output: {label}")
    print(f"Final Decision: {final_label}")
    print(f"Confidence: {score:.2f}")
    print("-" * 50)