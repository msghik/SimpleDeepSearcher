import json
import os
from urllib.parse import urlparse


CREDIBILITY_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'website_credibility.json')


def _domain(url: str) -> str:
    parsed = urlparse(url if url.startswith('http') else f'http://{url}')
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain


def load_scores(path: str = CREDIBILITY_FILE):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_scores(data, path: str = CREDIBILITY_FILE):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_score(url: str, data=None) -> int:
    domain = _domain(url)
    data = data or load_scores()
    for item in data:
        if _domain(item['URL']) == domain:
            return item.get('Credibility Score (/100)', 0)
    return 0


def mark_results(results, data=None, url_key='url', score_key='credibility'):
    data = data or load_scores()
    for r in results:
        url = r.get(url_key, '')
        r[score_key] = get_score(url, data)
    return results


def adjust_score(url: str, delta: int = -1, path: str = CREDIBILITY_FILE):
    data = load_scores(path)
    domain = _domain(url)
    for item in data:
        if _domain(item['URL']) == domain:
            item['Credibility Score (/100)'] = max(0, item.get('Credibility Score (/100)', 0) + delta)
            break
    else:
        data.append({
            'Rank': len(data) + 1,
            'Website Name': domain,
            'URL': domain,
            'Credibility Score (/100)': max(0, delta),
            'Primary Focus/Type': '',
            'Key Strengths Summary': ''
        })
    save_scores(data, path)
