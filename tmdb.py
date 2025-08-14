import os
import json
import requests

API_KEY = os.getenv("TMDB_API_KEY")  # Render ya da lokal .env'den gelir
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"


def _poster_url(path: str | None) -> str:
    return f"{POSTER_BASE}{path}" if path else ""


def search_movie(query: str):
    """TMDB'de film ara (id = tmdb{number}). IMDb/RT puanı eklemiyoruz; sonradan alınacak."""
    if not API_KEY:
        return []
    url = f"{BASE_URL}/search/movie"
    res = requests.get(url, params={"api_key": API_KEY, "query": query}).json()

    results = []
    for item in res.get("results", []):
        results.append({
            "id": f"tmdb{item.get('id')}",             # <- TMDB id'si
            "title": item.get("title") or "",
            "year": (item.get("release_date") or "")[:4] or "N/A",
            "poster": _poster_url(item.get("poster_path")),
            "description": item.get("overview", ""),
            "imdb": "",                                 # puanlar sonradan CSV/OMDb ile
            "rt": 0
        })
    return results


def search_tv(query: str):
    """TMDB'de dizi ara (id = tmdb{number})."""
    if not API_KEY:
        return []
    url = f"{BASE_URL}/search/tv"
    res = requests.get(url, params={"api_key": API_KEY, "query": query}).json()

    results = []
    for item in res.get("results", []):
        results.append({
            "id": f"tmdb{item.get('id')}",
            "title": item.get("name") or "",
            "year": (item.get("first_air_date") or "")[:4] or "N/A",
            "poster": _poster_url(item.get("poster_path")),
            "description": item.get("overview", ""),
            "imdb": "",
            "rt": 0
        })
    return results


def search_by_actor(actor_name: str):
    """
    Oyuncu adına göre arama yapar, TMDB 'person' sonucundaki known_for listesini
    film/dizi kartlarına dönüştürür. (id=tmdb{number}, media_type ekler.)
    """
    if not API_KEY:
        return []
    url = f"{BASE_URL}/search/person"
    res = requests.get(url, params={"api_key": API_KEY, "query": actor_name}).json()

    out = []
    for person in res.get("results", []):
        for work in person.get("known_for", []):
            media_type = work.get("media_type")  # "movie" | "tv"
            title = work.get("title") or work.get("name") or ""
            year = (work.get("release_date") or work.get("first_air_date") or "")[:4] or "N/A"
            out.append({
                "id": f"tmdb{work.get('id')}",
                "title": title,
                "year": year,
                "poster": _poster_url(work.get("poster_path")),
                "description": work.get("overview", ""),
                "imdb": "",
                "rt": 0,
                "media_type": media_type
            })
    return out


def add_to_favorites(item: dict, stars: int, media_type: str):
    """
    Yerel favorites.json'a ekler (Streamlit dışı basit kullanım için tutuluyor).
    Uygulamanın ana akışı Firestore yazıyor; bunu istersen kaldırabilirsin.
    """
    filename = "favorites.json"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):  # eski format desteği
                data = {"movies": data, "shows": []}
    except Exception:
        data = {"movies": [], "shows": []}

    key = "movies" if media_type == "movie" else "shows"
    item = dict(item)
    item["cineselectRating"] = stars
    data[key].append(item)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
