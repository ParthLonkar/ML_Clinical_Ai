import argparse

from model_utils import predict_with_explanation


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict triage class from clinical dialogue text.")
    parser.add_argument("--text", required=True, help="Dialogue text.")
    args = parser.parse_args()

    result = predict_with_explanation(args.text)

    print("Prediction")
    print(f"- label: {result['label']}")
    print(f"- confidence: {result['confidence']:.4f}")
    print(f"- uncertainty: {result['uncertainty']}")
    if result["rule_note"]:
        print(f"- rule: {result['rule_note']}")
    print(f"- suggestion_source: {result['suggestion_source']}")
    print(f"- suggestion: {result['suggestion']}")
    print("- class_probabilities:")
    for label, prob in sorted(result["class_probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {label}: {prob:.4f}")
    print("- top_evidence:")
    if result["top_evidence"]:
        for item in result["top_evidence"]:
            print(f"  - {item['token']}: {item['contribution']:.4f}")
    else:
        print("  - no strong positive token evidence found")
    print("- token_reading:")
    print(f"  - input_tokens: {', '.join(result['token_reading']['input_tokens']) if result['token_reading']['input_tokens'] else 'none'}")
    print(f"  - matched_tokens: {', '.join(result['token_reading']['matched_tokens']) if result['token_reading']['matched_tokens'] else 'none'}")
    print(f"  - match_coverage: {result['token_reading']['match_coverage']:.3f}")
    print("- similar_cases:")
    for case in result["similar_cases"]:
        matched = ", ".join(case["matched_tokens"]) if case["matched_tokens"] else "none"
        print(
            "  - "
            f"({case['label']}, sim={case['similarity']:.3f}, "
            f"vec={case['vector_similarity']:.3f}, tok={case['token_overlap']:.3f}, "
            f"matched=[{matched}]) {case['dialogue']}"
        )


if __name__ == "__main__":
    main()
