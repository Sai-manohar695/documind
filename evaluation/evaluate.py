import os
import sys
from dotenv import load_dotenv

# Add backend to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from qa_chain import ask_question

load_dotenv()

# ----------- Test Dataset -----------
# These are sample questions and expected answers
# based on a document you will upload for testing

test_cases = [
    {
        "question": "What is the main topic of this document?",
        "expected_keywords": ["main", "topic", "document"]
    },
    {
        "question": "Summarize the key points.",
        "expected_keywords": ["key", "points", "summary"]
    },
    {
        "question": "What conclusions are drawn?",
        "expected_keywords": ["conclusion", "result", "finding"]
    }
]


# ----------- Evaluation Metrics -----------

def keyword_coverage(answer: str, keywords: list) -> float:
    """Check what percentage of expected keywords appear in the answer."""
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 4) if keywords else 0.0


def evaluate_response(result: dict, expected_keywords: list) -> dict:
    """Evaluate a single Q&A response."""
    answer = result["answer"]
    confidence = result["confidence"]

    # Check if answer was refused due to low confidence
    is_refused = confidence["level"] == "very_low"

    # Keyword coverage score
    coverage = keyword_coverage(answer, expected_keywords)

    return {
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        "confidence_score": confidence["score"],
        "confidence_level": confidence["level"],
        "keyword_coverage": coverage,
        "was_refused": is_refused
    }


# ----------- Run Evaluation -----------

def run_evaluation():
    """Run all test cases and print evaluation report."""
    
    print("\n" + "="*60)
    print("         DOCUMIND EVALUATION REPORT")
    print("="*60)

    total_confidence = 0
    total_coverage = 0
    refused_count = 0

    for i, test in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {test['question']}")
        print("-" * 40)

        # Get answer from pipeline
        result = ask_question(query=test["question"])
        evaluation = evaluate_response(result, test["expected_keywords"])

        # Print results
        print(f"Answer Preview : {evaluation['answer_preview']}")
        print(f"Confidence     : {evaluation['confidence_level'].upper()} ({evaluation['confidence_score']})")
        print(f"Keyword Coverage: {evaluation['keyword_coverage'] * 100:.1f}%")
        print(f"Refused        : {'Yes' if evaluation['was_refused'] else 'No'}")

        total_confidence += evaluation["confidence_score"]
        total_coverage += evaluation["keyword_coverage"]
        if evaluation["was_refused"]:
            refused_count += 1

    # Summary
    n = len(test_cases)
    print("\n" + "="*60)
    print("                 SUMMARY")
    print("="*60)
    print(f"Total Tests         : {n}")
    print(f"Avg Confidence Score: {round(total_confidence / n, 4)}")
    print(f"Avg Keyword Coverage: {round(total_coverage / n * 100, 1)}%")
    print(f"Refused Answers     : {refused_count}/{n}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()