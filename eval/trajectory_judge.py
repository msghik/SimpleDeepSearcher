import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

PROMPT_TEMPLATE = (
    "You are a helpful assistant that evaluates question-answer pairs.\n"
    "Given a question, a reasoning trajectory, and a final answer, determine if the final answer correctly addresses the question."\
    " Respond with 'True' if the answer is correct or 'False' otherwise, then briefly explain why."
)


def format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    lines = []
    for step in trajectory:
        parts = [f"Step {step.get('step')}"]
        if 'thought' in step:
            parts.append(f"Thought: {step['thought']}")
        if 'action' in step:
            act = step['action']
            inp = step.get('action_input', '')
            parts.append(f"Action: {act} {inp}")
        if 'observation' in step:
            parts.append(f"Observation: {step['observation']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def evaluate(data: Dict[str, Any], model: str = "gpt-3.5-turbo") -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    trajectory_text = format_trajectory(data.get("trajectory", []))
    user_content = (
        f"Question: {data.get('question')}\n"
        f"Trajectory:\n{trajectory_text}\n"
        f"Answer: {data.get('answer')}"
    )
    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": user_content},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reasoning trajectory using OpenAI")
    parser.add_argument("input", help="Path to JSON file with question, trajectory and answer")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model name")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = evaluate(data, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
