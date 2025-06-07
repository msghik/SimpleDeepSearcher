import os
import sys
import requests
from inference.credibility import mark_results

persian_medical_thesaurus = {
    "فشار خون بالا": ["هایپرتانسیون", "پرفشاری خون", "افزایش فشار خون"],
    "دیابت": ["مرض قند", "افزایش قند خون", "دیابت شیرین", "دیابت نوع ۱", "دیابت نوع ۲"],
    "سرطان": ["نئوپلاسم", "تومور بدخیم", "کانسر", "کارسینوما"],
    "حمله قلبی": ["انفارکتوس میوکارد", "سکته قلبی", "MI", "ایست قلبی"],
    "سکته مغزی": ["حمله مغزی", "CVA", "استروک", "انسداد عروق مغزی"],
    "نارسایی کلیه": ["نارسایی مزمن کلیه", "CRF", "نارسایی حاد کلیه", "ازکارافتادگی کلیه"],
    "زخم معده": ["اولسر پپتیک", "زخم گوارشی", "زخم اثنی‌عشر"],
    "آسم": ["تنگی نفس", "بیماری انسدادی تنفسی", "حملات تنفسی", "آسم برونشیتی"],
    "کم‌خونی": ["آنمی", "کاهش هموگلوبین", "فقر آهن", "آنمی فقر آهن"],
    "صرع": ["تشنج", "حمله صرعی", "اپی‌لپسی"],
    "چربی خون بالا": ["هایپرلیپیدمی", "افزایش چربی خون", "کلسترول بالا"],
    "التهاب مفاصل": ["آرتریت", "ورم مفاصل", "آرتروز", "آرتریت روماتوئید"],
    "سرماخوردگی": ["نزدیک بینی", "عفونت ویروسی تنفسی", "رینوفارینژیت"],
    "سوزاک": ["گنوره", "عفونت گنوکوکی", "بیماری آمیزشی"],
    "ایدز": ["HIV", "عفونت HIV", "ویروس نقص ایمنی انسانی", "سندرم نقص ایمنی اکتسابی"]
}


def bing_search(query: str, api_key: str, count: int = 5):
    """Call the Bing Web Search API and return title and url for results."""
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count}
    response = requests.get(endpoint, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({"name": item.get("name"), "url": item.get("url")})
    return results


def search_term(term: str, api_key: str = None, count: int = 5):
    """Search Bing for a term using synonyms from the thesaurus if available."""
    if api_key is None:
        api_key = os.getenv("BING_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("Bing Search API key is required")

    synonyms = persian_medical_thesaurus.get(term)
    query_terms = synonyms if synonyms is not None else [term]
    query = " ".join(query_terms)
    results = bing_search(query, api_key, count)
    mark_results(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bing_persian_medical_search.py <term>")
        sys.exit(1)
    for result in search_term(sys.argv[1]):
        print(f"{result['name']}: {result['url']}")
