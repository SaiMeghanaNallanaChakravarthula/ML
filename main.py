import json
from difflib import get_close_matches
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"].lower() == question.lower():
            return q["answer"]

def chatbot():
    knowledge_base: dict = load_knowledge_base(r'D:\Sem 1\ML\knowledge_base.json')
    ground_truth_labels = []
    predicted_labels = []

    while True:
        user_input: str = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        ground_truth_labels.append(user_input.lower())

        best_match: str | None = find_best_match(user_input, [q["question"].lower() for q in knowledge_base["questions"]])
        predicted_labels.append(best_match.lower() if best_match else "unknown")

        if best_match:
            answer: str = get_answer_for_question(best_match, knowledge_base)
            print(f"Chatbot: {answer}")
        else:
            print("Chatbot: I don't know the answer. Can you teach me?")
            new_answer: str = input("Type the answer or 'skip' to skip: ")
            if new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base(r'D:\Sem 1\ML\knowledge_base.json', knowledge_base)
            print("Chatbot: Thank you for teaching me!")

    # Calculate and print confusion matrix
    unique_labels = np.unique(ground_truth_labels + ["unknown"])
    print("\nConfusion Matrix:")
    cm = confusion_matrix(ground_truth_labels, predicted_labels, labels=unique_labels)
    print(cm)

    # Calculate and print accuracy, precision, and recall
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
    recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    chatbot()
