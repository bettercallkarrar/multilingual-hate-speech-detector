def convert_to_hate_label(model_label):
    """
    Converts sentiment output into hate speech decision.

    Negative sentiment is mapped to Hate.
    Positive or neutral sentiment is mapped to Not Hate.
    """
    label = model_label.lower()

    if "negative" in label:
        return "Hate ❌"
    else:
        return "Not Hate ✅"