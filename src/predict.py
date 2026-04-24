from transformers import pipeline


def load_model():
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis"
    )


def convert_to_hate_label(model_label):
    label = model_label.lower()

    if "negative" in label:
        return "Hate"
    else:
        return "Not Hate"


def predict_text(text):
    classifier = load_model()
    result = classifier(text)[0]

    model_output = result["label"]
    confidence = result["score"]
    final_decision = convert_to_hate_label(model_output)

    return {
        "text": text,
        "model_output": model_output,
        "final_decision": final_decision,
        "confidence": confidence
    }


if __name__ == "__main__":
    text = input("Enter text: ")
    prediction = predict_text(text)

    print("\nPrediction Result")
    print("-" * 40)
    print(f"Text: {prediction['text']}")
    print(f"Model Output: {prediction['model_output']}")
    print(f"Final Decision: {prediction['final_decision']}")
    print(f"Confidence: {prediction['confidence']:.2f}")