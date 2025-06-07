import argparse
import json
from typing import List

from openai import OpenAI

from google_search import (
    google_web_search,
    extract_relevant_info,
    fetch_page_content,
)


PROMPT_TEMPLATE = "Summarize the following medical text:\n{text}\n"


def summarize_with_model(model_url: str, model_path: str, text: str) -> str:
    """Summarize text using an OpenAI-compatible endpoint."""
    client = OpenAI(base_url=model_url, api_key="EMPTY")
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}]
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_path,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Summarization error: {exc}")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize medical queries using Gemma models")
    parser.add_argument("--query", required=True, help="Medical question to search for")
    parser.add_argument("--subscription_key", required=True, help="Google search API key")
    parser.add_argument("--endpoint", default="https://google.serper.dev/search", help="Search API endpoint")
    parser.add_argument("--medgemma_url", required=True, help="Base url of MedGemma model")
    parser.add_argument("--medgemma_path", required=True, help="Model id for MedGemma")
    parser.add_argument("--gemma_url", required=True, help="Base url of Gemma model")
    parser.add_argument("--gemma_path", required=True, help="Model id for Gemma")
    parser.add_argument("--top_k", type=int, default=3, help="Number of search results to summarize")
    args = parser.parse_args()

    # Search the web
    search_results = google_web_search(args.query, args.subscription_key, args.endpoint)
    relevant = extract_relevant_info(search_results)[: args.top_k]
    urls = [item["url"] for item in relevant]

    # Fetch page contents
    contents = fetch_page_content(urls)

    # Summarize each document with MedGemma
    intermediate_summaries: List[str] = []
    for url in urls:
        text = contents.get(url, "")
        summary = summarize_with_model(args.medgemma_url, args.medgemma_path, text)
        intermediate_summaries.append(f"Source: {url}\n{summary}")

    combined = "\n\n".join(intermediate_summaries)

    # Summarize the combined text with Gemma for final answer
    final_summary = summarize_with_model(args.gemma_url, args.gemma_path, combined)

    output = {"query": args.query, "summary": final_summary, "sources": urls}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
