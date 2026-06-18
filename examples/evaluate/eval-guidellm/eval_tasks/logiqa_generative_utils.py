def doc_to_text(doc) -> str:
    if "query" in doc:
        return doc["query"] + " "

    choices = ["A", "B", "C", "D"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice}. {option}\n"
    prompt += "Answer with only one letter, A, B, C, or D.\nAnswer:"
    return prompt


def doc_to_target(doc) -> str:
    if "gold" in doc:
        gold = doc["gold"]
        if isinstance(gold, list):
            gold = gold[0]
        return "ABCD"[int(gold)]

    choices = ["a", "b", "c", "d"]
    return choices[choices.index(doc["label"].strip())].upper()
