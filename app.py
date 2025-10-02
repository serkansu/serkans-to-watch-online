from tmdb import search_movie, search_tv, search_by_actor
from omdb import get_ratings
import csv
from pathlib import Path
import streamlit as st
import requests
import firebase_admin
import base64
from firebase_admin import credentials, firestore
import json
import os
import time
# --- Turkish month and day name mappings ---
TURKISH_MONTHS = {
    "January": "Ocak",
    "February": "Şubat",
    "March": "Mart",
    "April": "Nisan",
    "May": "Mayıs",
    "June": "Haziran",
    "July": "Temmuz",
    "August": "Ağustos",
    "September": "Eylül",
    "October": "Ekim",
    "November": "Kasım",
    "December": "Aralık",
}
TURKISH_DAYS = {
    "Monday": "Pazartesi",
    "Tuesday": "Salı",
    "Wednesday": "Çarşamba",
    "Thursday": "Perşembe",
    "Friday": "Cuma",
    "Saturday": "Cumartesi",
    "Sunday": "Pazar",
}

# Helper to format datetime in Turkish (date only, no time)
def format_turkish_datetime(dt):
    return dt.strftime("%d/%m/%y")

# --- Robust Turkish/ISO date parser ---
from datetime import datetime as _DT

def parse_turkish_or_iso_date(v):
    """Return a datetime for various stored date formats.
    Supports:
      - Already-a-datetime (incl. Firestore Timestamp / DatetimeWithNanoseconds)
      - Turkish formatted strings like "05 Eylül 2025 Cuma"
      - ISO strings like "2025-09-06" or "2025-09-06 08:44:00" or "2025-09-06T08:44:00Z"
    On failure returns datetime.min.
    """
    try:
        if not v:
            return _DT.min
        # If it's already datetime-like (not pure str), convert via isoformat for safety
        if not isinstance(v, str) and hasattr(v, "isoformat"):
            try:
                return _DT.fromisoformat(v.isoformat().replace("Z", ""))
            except Exception:
                # Best effort: cast to string and keep parsing below
                v = v.isoformat()
        s = str(v)
        # Replace Turkish month/day names with English for parsing
        for eng, tr in TURKISH_MONTHS.items():
            s = s.replace(tr, eng)
        for eng, tr in TURKISH_DAYS.items():
            s = s.replace(tr, eng)
        # Try multiple known formats
        for fmt in (
            "%d/%m/%y",  # Try short Turkish/ISO date format first
            "%d %B %Y %A",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return _DT.strptime(s, fmt)
            except Exception:
                pass
        # Final fallback: fromisoformat if possible
        try:
            return _DT.fromisoformat(s.replace("Z", ""))
        except Exception:
            return _DT.min
    except Exception:
        return _DT.min
# --- JSON export helpers: make Firestore timestamps serializable & strip non-export fields ---
from datetime import datetime

# --- JSON export helpers: make Firestore timestamps serializable & strip non-export fields ---
class _EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # Firestore Timestamp is a subclass of datetime (DatetimeWithNanoseconds)
        if isinstance(o, datetime):
            try:
                return o.isoformat()
            except Exception:
                return str(o)
        return super().default(o)

def _strip_non_export_fields(item: dict) -> dict:
    """Return a shallow copy without fields we don't want in favorites_stw.json."""
    cleaned = dict(item or {})
    # Firestore-only metadata should not go to the public JSON
    cleaned.pop("addedAt", None)
    return cleaned
# --- /JSON export helpers ---
# --- helpers: normalize title for equality checks ---
def _norm_title(t: str) -> str:
    return (t or "").strip().lower()
# ---------- Sorting helpers for Streamio export ----------
ROMAN_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}

def _roman_to_int(s: str) -> int | None:
    s = (s or "").strip().lower()
    return ROMAN_MAP.get(s)

import re
_FRANCHISE_WORDS = {"the", "a", "an"}

def _normalize_franchise(title: str) -> str:
    """Get a coarse franchise/base name from a movie title.
    Examples:
      - "The Terminator" -> "terminator"
      - "Terminator 2: Judgment Day" -> "terminator"
      - "Back to the Future Part II" -> "back to the future"
    This is a heuristic; it deliberately keeps it simple.
    """
    t = (title or "").lower()
    # drop leading article
    parts = t.split()
    if parts and parts[0] in _FRANCHISE_WORDS and len(parts) > 1:
        t = " ".join(parts[1:])
    # keep text before a colon if it looks like a subtitle
    t = t.split(":")[0]
    # remove trailing sequel tokens like numbers/roman/"part X"
    t = re.sub(r"\bpart\s+[ivx]+\b", "", t).strip()
    t = re.sub(r"\bpart\s+\d+\b", "", t).strip()
    t = re.sub(r"\b\d+\b", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def _parse_sequel_number(title: str) -> int:
    """Try to extract sequel ordering number from a title.
    Returns 0 if not detected (so originals come first).
    Supports digits and roman numerals after words like 'part' or alone (e.g., 'Terminator 2').
    """
    t = (title or "").lower()
    # "Part II" / "Part 2"
    m = re.search(r"\bpart\s+([ivx]+|\d+)\b", t)
    if m:
        token = m.group(1)
        if token.isdigit():
            return int(token)
        ri = _roman_to_int(token)
        if ri:
            return ri
    # lone digits after the base word: e.g., "Terminator 2"
    m = re.search(r"\b(\d+)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # "II", "III" as standalone
    m = re.search(r"\b([ivx]{1,4})\b", t)
    if m:
        ri = _roman_to_int(m.group(1))
        if ri:
            return ri
    return 0

def _compute_franchise_min_year(items: list[dict]) -> dict[str, int]:
    """Return {base_name: min_year} for bases that have 2+ items in the list.
    Non-numeric/missing years are ignored.
    """
    years_by_base: dict[str, list[int]] = {}
    for it in items:
        base = _normalize_franchise(it.get("title", ""))
        try:
            y = int(it.get("year") or 0)
        except Exception:
            y = 0
        years_by_base.setdefault(base, []).append(y)
    return {b: min([y for y in ys if isinstance(y, int)]) for b, ys in years_by_base.items() if len(ys) >= 2}

def sort_media_for_export(items: list[dict], apply_franchise: bool = True) -> list[dict]:
    """Sort newest->oldest by *group year* (franchise min-year if grouped),
    then by sequel number (1,2,3…) inside the same franchise, otherwise by CineSelect.
    """
    items = list(items or [])
    base_min_year = _compute_franchise_min_year(items) if apply_franchise else {}

    def keyfn(it: dict):
        # group year: min franchise year if franchise exists (2+ items), else own year
        base = _normalize_franchise(it.get("title", ""))
        try:
            own_year = int(it.get("year") or 0)
        except Exception:
            own_year = 0
        group_year = base_min_year.get(base, own_year)
        # sequel number only meaningful if multiple in same base
        sequel_no = _parse_sequel_number(it.get("title", "")) if base in base_min_year else 0
        # tie-breaker by CineSelect rating (desc)
        cs = it.get("cineselectRating") or 0
        return (-group_year, base, sequel_no, -int(cs))

    return sorted(items, key=keyfn)
# ---------- /sorting helpers ----------
# --- seed_ratings.csv için yol ve ekleme fonksiyonu ---
SEED_PATH = Path(__file__).parent / "seed_ratings.csv"

def append_seed_rating(imdb_id, title, year, imdb_rating, rt_score):
    """seed_ratings.csv'ye (yoksa) yeni satır ekler; varsa dokunmaz."""
    if not imdb_id or imdb_id == "tt0000000":
        return

    # Zaten var mı kontrol et
    exists = False
    if SEED_PATH.exists():
        with SEED_PATH.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("imdb_id") == imdb_id:
                    exists = True
                    break
    if exists:
        return  # Aynı imdb_id zaten kayıtlı

    # Başlık yazmak gerekir mi?
    write_header = not SEED_PATH.exists() or SEED_PATH.stat().st_size == 0

    with SEED_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["imdb_id", "title", "year", "imdb_rating", "rt"])
        w.writerow([
            imdb_id,
            title,
            str(year or ""),
            (imdb_rating if imdb_rating is not None else ""),
            (rt_score if rt_score is not None else ""),
        ])
# --- /seed ekleme fonksiyonu ---

# --- seed okuma fonksiyonu ---
def read_seed_rating(imdb_id: str):
    """seed_ratings.csv içinden imdb_id ile eşleşen satırı döndürür.
    {'imdb_rating': float|None, 'rt': int|None} şeklinde veri verir; bulunamazsa None döner.
    Hem 'imdb_id' hem de 'imdb' sütun adlarını destekler.
    """
    try:
        iid = (imdb_id or "").strip()
        if not iid or not SEED_PATH.exists():
            return None
        with SEED_PATH.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                key = (row.get("imdb_id") or row.get("imdb") or "").strip()
                if key == iid:
                    # değerleri temizle
                    ir = row.get("imdb_rating")
                    rt = row.get("rt")
                    try:
                        ir_val = float(ir) if ir not in (None, "", "N/A") else None
                    except Exception:
                        ir_val = None
                    try:
                        rt_val = int(float(rt)) if rt not in (None, "", "N/A") else None
                    except Exception:
                        rt_val = None
                    return {"imdb_rating": ir_val, "rt": rt_val}
    except Exception:
        pass
    return None
# --- /seed okuma fonksiyonu ---
def get_ratings(imdb_id):
    import requests
    OMDB_API_KEY = os.getenv("OMDB_API_KEY")
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
    except Exception as e:
        return {"imdb_rating": None, "rt": None, "error": str(e)}

    imdb_rating = None
    rt_score = None

    # IMDb rating doğrudan al
    if "imdbRating" in data and data["imdbRating"] != "N/A":
        try:
            imdb_rating = float(data["imdbRating"])
        except Exception:
            imdb_rating = None

    # Rotten Tomatoes puanını Ratings array’den al
    if "Ratings" in data:
        for rating in data["Ratings"]:
            if rating["Source"] == "Rotten Tomatoes":
                try:
                    rt_score = int(rating["Value"].replace("%", ""))
                except Exception:
                    rt_score = None
                break

    return {"imdb_rating": imdb_rating, "rt": rt_score}

def get_imdb_id_from_tmdb(title, year=None, is_series=False):
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if not tmdb_api_key:
        print("❌ TMDB API key not found in environment variables.")
        return ""

    search_type = "tv" if is_series else "movie"
    search_url = f"https://api.themoviedb.org/3/search/{search_type}"
    params = {
        "api_key": tmdb_api_key,
        "query": title,
        "year": year if not is_series else None,
        "first_air_date_year": year if is_series else None,
    }

    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        return ""

    results = response.json().get("results", [])
    if not results:
        return ""

    tmdb_id = results[0]["id"]
    external_ids_url = f"https://api.themoviedb.org/3/{search_type}/{tmdb_id}/external_ids"
    external_response = requests.get(external_ids_url, params={"api_key": tmdb_api_key})
    if external_response.status_code != 200:
        return ""

    imdb_id = external_response.json().get("imdb_id", "")
    return imdb_id or ""


# --- NEW: search_by_director_writer helper ---
def search_by_director_writer(name: str) -> list[dict]:
    """
    Given a person name, return movies/TV shows where they worked in Directing or Writing (incl. 'Creator').
    Results contain: id, title, year, poster_path, poster (full URL), media_type ('movie'|'tv').
    """
    try:
        tmdb_api_key = os.getenv("TMDB_API_KEY")
        if not tmdb_api_key or not (name or "").strip():
            return []

        # 1) Find the person by name
        r = requests.get(
            "https://api.themoviedb.org/3/search/person",
            params={"api_key": tmdb_api_key, "query": name, "include_adult": False},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        people = (r.json() or {}).get("results", []) or []
        if not people:
            return []
        person_id = people[0].get("id")
        if not person_id:
            return []

        # 2) Fetch their combined credits
        cr = requests.get(
            f"https://api.themoviedb.org/3/person/{person_id}/combined_credits",
            params={"api_key": tmdb_api_key},
            timeout=20,
        )
        if cr.status_code != 200:
            return []
        data = cr.json() or {}
        crew_list = data.get("crew", []) or []

        allowed_depts = {"Directing", "Writing"}
        allowed_jobs = {"Director", "Writer", "Screenplay", "Story", "Creator", "Developed by", "Showrunner"}

        # Deduplicate by (media_type, tmdb_id)
        seen = set()
        out: list[dict] = []
        for c in crew_list:
            dept = (c.get("department") or "").strip()
            job = (c.get("job") or "").strip()
            if not ((dept in allowed_depts) or (job in allowed_jobs)):
                continue

            media_type = c.get("media_type") or ("tv" if c.get("first_air_date") else "movie")
            tmdb_id = c.get("id")
            if tmdb_id is None:
                continue
            key = f"{media_type}:{tmdb_id}"
            if key in seen:
                continue
            seen.add(key)

            title = c.get("title") if media_type == "movie" else (c.get("name") or c.get("title") or "")
            date = c.get("release_date") if media_type == "movie" else c.get("first_air_date")
            try:
                year = int(str(date)[:4]) if date else None
            except Exception:
                year = None
            poster_path = c.get("poster_path") or ""

            out.append({
                "id": str(tmdb_id),
                "title": title,
                "year": year,
                "poster_path": poster_path,
                "poster": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "",
                "media_type": media_type,
            })

        return out
    except Exception:
        return []
def push_favorites_to_github():
    """Push favorites.json and seed_ratings.csv to their respective GitHub repos.
    - favorites.json  -> serkansu/serkans-to-watch-addon
    - seed_ratings.csv -> serkansu/serkans-to-watch-online
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        st.warning("⚠️ GITHUB_TOKEN environment variable is missing!")
        st.error("❌ GitHub token bulunamadı. Environment variable ayarlanmalı.")
        return

    # Which file goes to which repo
    publish_plan = [
        {"file": "favorites_stw.json", "owner": "serkansu", "repo": "serkans-to-watch-addon"},
        {"file": "seed_ratings.csv", "owner": "serkansu", "repo": "serkans-to-watch-online"},
    ]

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    for item in publish_plan:
        file_path = item["file"]
        repo_owner = item["owner"]
        repo_name = item["repo"]
        commit_message = f"Update {file_path} via Streamlit sync"
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

        # Read file to upload; skip if missing
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except FileNotFoundError:
            st.warning(f"⚠️ Dosya bulunamadı, atlandı: {file_path}")
            continue

        encoded_content = base64.b64encode(content).decode("utf-8")

        # Get current SHA if file exists
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            sha = response.json().get("sha")
        elif response.status_code == 404:
            sha = None
        else:
            st.error(f"❌ GitHub API erişim hatası ({file_path} → {repo_owner}/{repo_name}): {response.status_code}")
            try:
                st.code(response.json())
            except Exception:
                pass
            continue

        payload = {
            "message": commit_message,
            "content": encoded_content,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha

        put_response = requests.put(url, headers=headers, json=payload)
        if put_response.status_code not in (200, 201):
            st.error(f"❌ Push başarısız ({file_path} → {repo_owner}/{repo_name}): {put_response.status_code}")
            try:
                st.code(put_response.json())
            except Exception:
                pass
        else:
            st.success(f"✅ Push OK: {file_path} → {repo_owner}/{repo_name}")
import streamlit as st
from firebase_setup import get_firestore

def fix_invalid_imdb_ids(data):
    for section in ["movies", "shows"]:
        for item in data[section]:
            if isinstance(item.get("imdb"), (int, float)):
                item["imdb"] = ""

def sort_flat_for_export(items, mode):
    """Sort a flat media list by the selected mode.
    modes:
      - 'cc'   : CineSelect DESC (highest first; ties -> IMDb DESC, then Year DESC)
      - 'imdb' : IMDb DESC
      - 'year' : Year DESC
    """
    def key_fn(it):
        if mode == "cc":
            # CineSelect: descending (highest first); tie-break IMDb (desc), then year (desc)
            try:
                cs = int(it.get("cineselectRating") or 0)
            except Exception:
                cs = 0
            try:
                imdb = float(it.get("imdbRating") or 0)
            except Exception:
                imdb = 0.0
            try:
                year = int(str(it.get("year", "0")).strip() or 0)
            except Exception:
                year = 0
            # For reverse=True (see return), sort by (cs, imdb, year) descending
            return (cs, imdb, year)
        elif mode == "imdb":
            v = it.get("imdbRating")
            try:
                return float(v) if v not in (None, "", "N/A") else -1
            except Exception:
                return -1
        elif mode == "year":
            try:
                return int(str(it.get("year", "0")).strip() or 0)
            except Exception:
                return 0
        # default -> behave like 'cc'
        try:
            cs = int(it.get("cineselectRating") or 0)
        except Exception:
            cs = 0
        try:
            imdb = float(it.get("imdbRating") or 0)
        except Exception:
            imdb = 0.0
        try:
            year = int(str(it.get("year", "0")).strip() or 0)
        except Exception:
            year = 0
        return (cs, imdb, year)

    # cc -> descending (reverse=True); imdb/year -> descending (reverse=True)
    return sorted(items or [], key=key_fn, reverse=True)

# ---------------------- CineSelect clamp & sync helpers ----------------------
def _clamp_cs(v: int | float) -> int:
    try:
        iv = int(v)
    except Exception:
        iv = 0
    if iv < 1:
        return 1
    if iv > 100:
        return 100
    return iv

# Streamlit on_change helpers to keep slider and input in sync

def _sync_cs_from_slider(src_key: str, dst_key: str):
    v = _clamp_cs(st.session_state.get(src_key, 0))
    st.session_state[src_key] = v
    st.session_state[dst_key] = v


def _sync_cs_from_input(src_key: str, dst_key: str):
    v = _clamp_cs(st.session_state.get(src_key, 0))
    st.session_state[src_key] = v
    st.session_state[dst_key] = v


# --- Safe session state setter for Streamlit widgets ---

def _safe_set_state(key: str, value):
    """Set session_state[key] safely; ignore Streamlit exceptions that occur
    when attempting to modify a widget key after instantiation within the same run."""
    try:
        st.session_state[key] = value
    except Exception:
        # On Streamlit rerun, the new value will be reflected anyway.
        pass

# --- Simple session-based auth gate using an env var ---
def ensure_authenticated():
    """If APP_ACCESS_KEY is set in env, ask for it once per browser tab.
    When correct, remember in st.session_state until the tab is closed or refreshed.
    """
    key = (os.getenv("APP_ACCESS_KEY") or "").strip()
    if not key:
        # No key configured -> app stays public
        return

    if st.session_state.get("_auth_ok", False):
        return

    st.title("🔒 Serkan’s To‑Watch Online")
    st.info("Bu sayfa şifre ile korunuyor. Lütfen erişim anahtarını girin.")
    pw = st.text_input("Şifre", type="password", key="__app_pw")
    if st.button("Giriş", key="__app_login"):
        if pw == key:
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Yanlış şifre. Tekrar deneyin.")
    st.stop()
# --- /auth gate ---

def sync_with_firebase(sort_mode="imdb"):
    favorites_data = {
        "movies": st.session_state.get("favorite_movies", []),
        "shows": st.session_state.get("favorite_series", [])
    }
    fix_invalid_imdb_ids(favorites_data)  # IMDb puanı olanları temizle
    # Normalize type after IMDb correction
    for section in ["movies", "shows"]:
        for item in favorites_data[section]:
            t = (item.get("type") or "").lower()
            if t in ["tv", "tvshow", "show", "series"]:
                item["type"] = "show"
            elif t in ["movie", "film"]:
                item["type"] = "movie"
# IMDb ID eksikse ➜ tamamlama başlıyor
        # Eksik imdb id'leri tamamla
    for section in ["movies", "shows"]:
        for item in favorites_data[section]:
            if not item.get("imdb") or item.get("imdb") == "":
                title = item.get("title")
                year = item.get("year")
                raw_type = item.get("type", "").lower()
                section_name = section.lower()

                is_series_by_section = section_name in ["shows", "series"]
                is_series_by_type = raw_type in ["series", "tv", "tv_show", "tvshow", "show"]

                is_series = is_series_by_section or is_series_by_type
                # NOTE: İç tip alanını tutarlı hale getiriyoruz: dizi için 'show', film için 'movie'
                item["type"] = "show" if is_series else "movie"
                imdb_id = get_imdb_id_from_tmdb(title, year, is_series=is_series)
                # IMDb ve RT puanlarını çek
                stats = get_ratings(imdb_id)
                imdb_rating = stats.get("imdb_rating") if stats else None
                rt_score = stats.get("rt") if stats else None
                print(f"🎬 {title} ({year}) | is_series={is_series} → IMDb ID: {imdb_id}")
                item["imdb"] = imdb_id
                item["imdbRating"] = float(imdb_rating) if imdb_rating is not None else 0.0
                item["rt"] = int(rt_score) if rt_score is not None else 0
                # ⬇️ YENİ: seed_ratings.csv’ye (yoksa) ekle
                append_seed_rating(imdb_id, title, year, imdb_rating, rt_score)
    # seed_ratings.csv içinde her favorinin olduğundan emin ol (CSV'de zaten varsa eklenmez)
    for _section in ("movies", "shows"):
        for _it in favorites_data.get(_section, []):
            append_seed_rating(
                imdb_id=_it.get("imdb"),
                title=_it.get("title"),
                year=_it.get("year"),
                imdb_rating=_it.get("imdbRating"),
                rt_score=_it.get("rt"),
            )
    # ---- Filter out watched items (only export "to_watch" items, treat missing/blank status as to_watch)
    movies_to_export = [x for x in favorites_data.get("movies", []) if (x.get("status") in (None, "", "to_watch"))]
    shows_to_export  = [x for x in favorites_data.get("shows",  []) if (x.get("status") in (None, "", "to_watch"))]

    # ---- Apply export ordering
    sorted_movies = sort_flat_for_export(movies_to_export, sort_mode)
    sorted_series = sort_flat_for_export(shows_to_export, sort_mode)

    # Remove Firestore-only fields and ensure JSON-serializable types (timestamps -> ISO strings)
    export_movies = [_strip_non_export_fields(x) for x in sorted_movies]
    export_series = [_strip_non_export_fields(x) for x in sorted_series]

    # Dışarı yazarken anahtar adını 'shows' -> 'series' olarak çevir
    output_data = {
        "movies": export_movies,
        "series": export_series,
    }
    with open("favorites_stw.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, cls=_EnhancedJSONEncoder)
        st.write("🔍 FAVORITES DEBUG (output):", output_data)
    st.success("✅ favorites_stw.json dosyası yerel olarak oluşturuldu.")

    # GitHub'a push et
    push_favorites_to_github()


# --- New: Sync Watched with Firebase/GitHub for watched items only ---
def sync_watched_with_firebase(sort_mode="imdb"):
    """Export watched movies/series to favorites_watched.json and push to serkansu/serkan-watched-addon."""
    favorites_data = {
        "movies": st.session_state.get("favorite_movies", []),
        "shows": st.session_state.get("favorite_series", [])
    }
    fix_invalid_imdb_ids(favorites_data)
    # Normalize type after IMDb correction
    for section in ["movies", "shows"]:
        for item in favorites_data[section]:
            t = (item.get("type") or "").lower()
            if t in ["tv", "tvshow", "show", "series"]:
                item["type"] = "show"
            elif t in ["movie", "film"]:
                item["type"] = "movie"
    # Eksik imdb id'leri tamamla (same as in sync_with_firebase)
    for section in ["movies", "shows"]:
        for item in favorites_data[section]:
            if not item.get("imdb") or item.get("imdb") == "":
                title = item.get("title")
                year = item.get("year")
                raw_type = item.get("type", "").lower()
                section_name = section.lower()
                is_series_by_section = section_name in ["shows", "series"]
                is_series_by_type = raw_type in ["series", "tv", "tv_show", "tvshow", "show"]
                is_series = is_series_by_section or is_series_by_type
                item["type"] = "show" if is_series else "movie"
                imdb_id = get_imdb_id_from_tmdb(title, year, is_series=is_series)
                stats = get_ratings(imdb_id)
                imdb_rating = stats.get("imdb_rating") if stats else None
                rt_score = stats.get("rt") if stats else None
                item["imdb"] = imdb_id
                item["imdbRating"] = float(imdb_rating) if imdb_rating is not None else 0.0
                item["rt"] = int(rt_score) if rt_score is not None else 0
                append_seed_rating(imdb_id, title, year, imdb_rating, rt_score)
    # seed_ratings.csv içinde her favorinin olduğundan emin ol (CSV'de zaten varsa eklenmez)
    for _section in ("movies", "shows"):
        for _it in favorites_data.get(_section, []):
            append_seed_rating(
                imdb_id=_it.get("imdb"),
                title=_it.get("title"),
                year=_it.get("year"),
                imdb_rating=_it.get("imdbRating"),
                rt_score=_it.get("rt"),
            )
    # ---- Only export watched items (status == "watched")
    movies_to_export = [x for x in favorites_data.get("movies", []) if x.get("status") == "watched"]
    shows_to_export  = [x for x in favorites_data.get("shows",  []) if x.get("status") == "watched"]
    # ---- Sort watched items by watched date descending (ignore sort_mode)
    sorted_movies = sorted(movies_to_export, key=lambda it: parse_turkish_or_iso_date(it.get("watchedAt")), reverse=True)
    sorted_series = sorted(shows_to_export, key=lambda it: parse_turkish_or_iso_date(it.get("watchedAt")), reverse=True)
    export_movies = [_strip_non_export_fields(x) for x in sorted_movies]
    export_series = [_strip_non_export_fields(x) for x in sorted_series]
    output_data = {
        "movies": export_movies,
        "series": export_series,
    }
    with open("favorites_watched.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, cls=_EnhancedJSONEncoder)
        st.write("🔍 FAVORITES WATCHED DEBUG (output):", output_data)
    st.success("✅ favorites_watched.json dosyası yerel olarak oluşturuldu.")

    # Push to the correct GitHub repo (serkansu/serkan-watched-addon)
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        st.warning("⚠️ GITHUB_TOKEN environment variable is missing!")
        st.error("❌ GitHub token bulunamadı. Environment variable ayarlanmalı.")
        return
    file_path = "favorites_watched.json"
    repo_owner = "serkansu"
    repo_name = "serkan-watched-addon"
    commit_message = f"Update {file_path} via Streamlit sync (watched)"
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        st.warning(f"⚠️ Dosya bulunamadı, atlandı: {file_path}")
        return
    encoded_content = base64.b64encode(content).decode("utf-8")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        sha = response.json().get("sha")
    elif response.status_code == 404:
        sha = None
    else:
        st.error(f"❌ GitHub API erişim hatası ({file_path} → {repo_owner}/{repo_name}): {response.status_code}")
        try:
            st.code(response.json())
        except Exception:
            pass
        return
    payload = {
        "message": commit_message,
        "content": encoded_content,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    put_response = requests.put(url, headers=headers, json=payload)
    if put_response.status_code not in (200, 201):
        st.error(f"❌ Push başarısız ({file_path} → {repo_owner}/{repo_name}): {put_response.status_code}")
        try:
            st.code(put_response.json())
        except Exception:
            pass
    else:
        st.success(f"✅ Push OK: {file_path} → {repo_owner}/{repo_name}")

# Firestore will be initialized AFTER auth gate below.

# --- Inserted: Page config and auth gate ---
st.set_page_config(page_title="Serkan’s To‑Watch Online", page_icon="🍿", layout="wide")
ensure_authenticated()
# --- Global CSS for input/text color (theme-adaptive) ---
st.markdown(
    """
    <style>
    textarea, input, .stTextInput, .stTextArea {
        color: inherit !important;
        background-color: inherit !important;
    }
    .comment-box {
        color: #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Firestore'dan verileri çek ve session'a yaz (only after auth)
db = get_firestore()
# Sadece ilk yüklemede Firestore'dan getir (cache yok)
if "favorite_movies" not in st.session_state or "favorite_series" not in st.session_state:
    movies = [doc.to_dict() for doc in db.collection("favorites").where("type", "in", ["movie", "film"]).stream()]
    shows = [doc.to_dict() for doc in db.collection("favorites").where("type", "in", ["show", "series", "tv", "tvshow"]).stream()]
    st.session_state["favorite_movies"] = movies
    st.session_state["favorite_series"] = shows

# --- Mobile Home Screen & Favicons ---
# High-res icons for iOS/Android home screen shortcuts and browser favicons.
ICON_180 = "https://em-content.zobj.net/source/apple/391/popcorn_1f37f.png"      # iOS (180x180)
ICON_192 = "https://em-content.zobj.net/source/microsoft-teams/363/popcorn_1f37f.png"  # Android (~192x192)
ICON_512 = "https://em-content.zobj.net/source/telegram/358/popcorn_1f37f.png"   # Android (~512x512)

st.markdown(
    f"""
    <link rel="apple-touch-icon" sizes="180x180" href="{ICON_180}">
    <link rel="icon" type="image/png" sizes="192x192" href="{ICON_192}">
    <link rel="icon" type="image/png" sizes="512x512" href="{ICON_512}">
    <meta name="theme-color" content="#111111">
    """,
    unsafe_allow_html=True,
)
# --- /Mobile Home Screen & Favicons ---


from datetime import datetime as _dt
import pytz
st.markdown("<h1>🍿 Serkan'ın İzlenecek Film & Dizi Listesi <span style='color: orange'>ONLINE ✅</span></h1>", unsafe_allow_html=True)
deploy_placeholder = st.empty()
deploy_placeholder.markdown(
    f"<div style='color: gray; font-size: 12px;'>Deployed: {_dt.now(pytz.timezone('Europe/Istanbul')).strftime('%d/%m/%Y %H:%M:%S')}</div>",
    unsafe_allow_html=True
)

# --- Favorites count helper (moved above usage) ---
def show_favorites_count():
    movie_docs = db.collection("favorites").where("type", "==", "movie").stream()
    series_docs = db.collection("favorites").where("type", "==", "show").stream()
    movie_count = len(list(movie_docs))
    series_count = len(list(series_docs))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎬 Favorite Movies", movie_count)
    with col2:
        st.metric("📺 Favorite TV Shows", series_count)

# --- Quick Toolbar (always visible) ---
# Ensure required defaults
if "show_posters" not in st.session_state:
    st.session_state["show_posters"] = True
if "sync_sort_mode" not in st.session_state:
    st.session_state["sync_sort_mode"] = "cc"

# -- Toolbar with new watched sync button --
tcol1, tcol2, tcol3, tcol4, tcol_sp = st.columns([1.2, 1.6, 1.8, 1.8, 4])
with tcol1:
    if st.button("🖼️ Toggle Posters", key="toolbar_toggle_posters"):
        st.session_state["show_posters"] = not st.session_state.get("show_posters", True)
with tcol2:
    if st.button("📂 JSON & CSV Sync", key="toolbar_sync"):
        sync_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "cc"))
        st.success("✅ favorites_stw.json ve seed_ratings.csv senkronize edildi.")
with tcol3:
    if st.button("📂 JSON & CSV Sync (Watched)", key="toolbar_sync_watched"):
        sync_watched_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "cc"))
        st.success("✅ favorites_watched.json senkronize edildi (Watched).")
with tcol4:
    if st.button("📊 Favori Sayıları", key="toolbar_counts"):
        show_favorites_count()



#
# --- Options Expander (All action buttons grouped) ---
with st.expander("✨ Options"):
    # 1. Go to Top
    if st.button("🏠 Go to Top"):
        st.rerun()

    # 2. Toggle Posters
    if "show_posters" not in st.session_state:
        _safe_set_state("show_posters", True)
    if st.button("🖼️ Toggle Posters"):
        st.session_state["show_posters"] = not st.session_state["show_posters"]

    # 3. JSON & CSV Sync
    if "sync_sort_mode" not in st.session_state:
        _safe_set_state("sync_sort_mode", "cc")
    if st.button("📂 JSON & CSV Sync"):
        sync_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "cc"))
        st.success("✅ favorites_stw.json ve seed_ratings.csv senkronize edildi.")
    # 3b. Watched Sync
    if st.button("📂 JSON & CSV Sync (Watched)"):
        sync_watched_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "cc"))
        st.success("✅ favorites_watched.json senkronize edildi (Watched).")
    st.radio(
        "Sync sıralaması",
        ["imdb", "cc", "year"],
        key="sync_sort_mode",
        horizontal=False,
        help="Tümü: yüksekten düşüğe (CineSelect, IMDb, Year)."
    )

    # 4. Favori Sayılarını Göster
    if st.button("📊 Favori Sayılarını Göster"):
        show_favorites_count()

show_posters = st.session_state["show_posters"]
media_type = st.radio("Search type:", ["Movie", "TV Show", "Actor/Actress", "Director/Writer"], horizontal=True)

# ---- Safe clear for search widgets (avoid modifying after instantiation)
if "clear_search" not in st.session_state:
    _safe_set_state("clear_search", False)

if st.session_state.clear_search:
    # reset the flag and clear both the input widget's value and the session copy
    st.session_state.clear_search = False
    _safe_set_state("query_input", "")
    _safe_set_state("query", "")

if "query" not in st.session_state:
    _safe_set_state("query", "")

query = st.text_input(
    f"🔍 Search for a {media_type.lower()}",
    key="query_input",
)
st.session_state.query = query

# --- Build quick lookups for existing favorites (to warn inside search results)
_current_sort = st.session_state.get("fav_sort", "CineSelect")
_movies_all = list(st.session_state.get("favorite_movies", []))
_shows_all  = list(st.session_state.get("favorite_series", []))


# --- Sorting key for favorites
def get_sort_key(fav):
    sort_name = st.session_state.get("fav_sort", "CineSelect")
    try:
        # Always return tuple: (CS, IMDb, Year) for CineSelect mode
        if sort_name == "CineSelect":
            cs = int(fav.get("cineselectRating") or 0)
            try:
                imdb = float(fav.get("imdbRating") or 0)
            except Exception:
                imdb = 0.0
            try:
                year = int(fav.get("year") or 0)
            except Exception:
                year = 0
            # For reverse=True, sort DESC
            return (cs, imdb, year)
        elif sort_name == "IMDb":
            return float(fav.get("imdbRating") or 0)
        elif sort_name == "RT":
            return float(fav.get("rt") or 0)
        elif sort_name == "Year":
            return int(fav.get("year") or 0)
    except Exception:
        return 0

# Reuse existing sorting logic so positions match the list below
_movies_sorted = sorted(_movies_all, key=get_sort_key, reverse=True)
_shows_sorted  = sorted(_shows_all, key=get_sort_key, reverse=True)

# Maps like "title::year" -> (favorite_dict, position)
_movies_idx = {}
for pos, f in enumerate(_movies_sorted, start=1):
    _key = f"{_norm_title(f.get('title'))}::{str(f.get('year') or '')}"
    _movies_idx[_key] = (f, pos)

_shows_idx = {}
for pos, f in enumerate(_shows_sorted, start=1):
    _key = f"{_norm_title(f.get('title'))}::{str(f.get('year') or '')}"
    _shows_idx[_key] = (f, pos)
if query:
    st.session_state.query = query
    if media_type == "Movie":
        results = search_movie(query)
    elif media_type == "TV Show":
        results = search_tv(query)
    elif media_type == "Actor/Actress":
        results = search_by_actor(query)
    else:  # "Director/Writer"
        results = search_by_director_writer(query)

    try:
        results = sorted(results, key=lambda x: x.get("cineselectRating", 0), reverse=True)
    except:
        pass

    if not results:
        st.error("❌ No results found.")
    else:
        # Sıralama: yıl'a göre yeni→eski
        results.sort(key=lambda x: x.get("year", 0), reverse=True)

        # Seçimler için checkbox listesi
        selections = []
        for idx, item in enumerate(results):
            st.divider()
            # 🔗 Normalize poster and ensure clickable IMDb link for search results
            if show_posters:
                poster_url = item.get("Poster") or item.get("poster") or item.get("poster_path") or ""
                # If we got a raw TMDb /poster_path, prefix the full TMDb image base
                if poster_url.startswith("/"):
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_url}"

                # Prefer existing IDs; if not present, resolve via TMDb external_ids
                imdb_id = item.get("imdbID") or item.get("imdb_id") or item.get("imdb") or ""
                if not imdb_id:
                    title_for_lookup = item.get("title") or item.get("Title") or ""
                    year_for_lookup = item.get("year")
                    try:
                        imdb_id = get_imdb_id_from_tmdb(
                            title_for_lookup,
                            year_for_lookup,
                            is_series=(media_type == "TV Show"),
                        ) or ""
                        # Cache back into the item so later steps (add_to_favorites) can reuse it
                        if imdb_id:
                            item["imdbID"] = imdb_id
                    except Exception:
                        imdb_id = ""

                imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else (item.get("imdb_url", "") or "")

                if poster_url:
                    if imdb_url:
                        st.markdown(
                            f"<a href='{imdb_url}' target='_blank'>"
                            f"<img src='{poster_url}' alt='{item.get('Title', item.get('title', ''))}' width='180'/>"
                            "</a>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.image(poster_url, width=180)

            # Checkbox ile seçim (tekli ekleme butonunu kaldırdık)
            checkbox_label = f"{item['title']} ({item.get('year','?')})"
            selected = st.checkbox(checkbox_label, key=f"select_{item.get('id',item['title'])}")
            if selected:
                selections.append(item)

            st.markdown(f"**{idx+1}. {item['title']} ({item.get('year', '—')})**")

            # --- warn inline if this exact title+year already exists in favorites ---
            _item_key = f"{_norm_title(item.get('title'))}::{str(item.get('year') or '')}"
            if media_type == "Movie" and _item_key in _movies_idx:
                _fav, _pos = _movies_idx[_item_key]
                _cs = _fav.get('cineselectRating', 'N/A')
                _added = _fav.get('addedAt')
                try:
                    _added_txt = format_turkish_datetime(_added) if hasattr(_added, "strftime") else str(_added) if _added else "—"
                except Exception:
                    _added_txt = "—"
                st.warning(f"⚠️ Dikkat: Bu film listende zaten var → sıra: #{_pos} • CS: {_cs} • eklenme: {_added_txt}", icon="⚠️")
            elif media_type == "TV Show" and _item_key in _shows_idx:
                _fav, _pos = _shows_idx[_item_key]
                _cs = _fav.get('cineselectRating', 'N/A')
                _added = _fav.get('addedAt')
                try:
                    _added_txt = format_turkish_datetime(_added) if hasattr(_added, "strftime") else str(_added) if _added else "—"
                except Exception:
                    _added_txt = "—"
                st.warning(f"⚠️ Dikkat: Bu dizi listende zaten var → sıra: #{_pos} • CS: {_cs} • eklenme: {_added_txt}", icon="⚠️")

            # IMDb rating display: prefer explicit imdbRating; if not present, use numeric `imdb` when it is a rating
            _imdb_rating_field = item.get("imdbRating", None)
            if isinstance(_imdb_rating_field, (int, float)):
                imdb_display = f"{float(_imdb_rating_field):.1f}" if _imdb_rating_field > 0 else "N/A"
            elif isinstance(item.get("imdb"), (int, float)):
                imdb_display = f"{float(item['imdb']):.1f}" if item["imdb"] > 0 else "N/A"
            else:
                imdb_display = "N/A"

            rt_val = item.get("rt", 0)
            rt_display = f"{int(rt_val)}%" if isinstance(rt_val, (int, float)) and rt_val > 0 else "N/A"
            st.markdown(f"⭐ IMDb: {imdb_display} &nbsp;&nbsp; 🍅 RT: {rt_display}", unsafe_allow_html=True)

            # --- Tekli ekleme butonu kaldırıldı ---
            # --- CineSelect manual input kaldırıldı ---

        # Toplu ekleme butonu
        def add_to_favorites(item, cs_score=101):
            # media_key: Movie/TV Show ayrımı (robust for mixed results)
            media_key = (
                "movie" if (item.get("media_type") == "movie" or media_type == "Movie")
                else "show" if (item.get("media_type") == "tv" or media_type == "TV Show" or item.get("first_air_date"))
                else "movie"
            )
            from omdb import get_ratings, fetch_ratings
            imdb_id = (item.get("imdb") or "").strip()
            if not imdb_id or imdb_id == "tt0000000":
                imdb_id = get_imdb_id_from_tmdb(
                    title=item["title"],
                    year=item.get("year"),
                    is_series=(media_key == "show"),
                )
            # IMDb/RT puanlarını getir (ÖNCE yerel CSV, yoksa OMDb-ID, o da yoksa Title/Year)
            stats = {}
            raw_id = {}
            raw_title = {}
            source = None
            seed_hit = read_seed_rating(imdb_id)
            if seed_hit and (seed_hit.get("imdb_rating") or seed_hit.get("rt")):
                stats = {"imdb_rating": seed_hit.get("imdb_rating"), "rt": seed_hit.get("rt")}
                source = "CSV"
            if not source:
                if imdb_id:
                    stats = get_ratings(imdb_id) or {}
                    raw_id = (stats.get("raw") or {})
                    source = "CSV/OMDb-ID" if raw_id else None
            if not stats or ((stats.get("imdb_rating") in (None, 0, "N/A")) and (stats.get("rt") in (None, 0, "N/A"))):
                ir, rt, raw_title = fetch_ratings(item["title"], item.get("year"))
                stats = {"imdb_rating": ir, "rt": rt}
                if not source:
                    source = "OMDb-title"
            imdb_rating = float(stats.get("imdb_rating") or 0.0)
            rt_score    = int(stats.get("rt") or 0)
            _doc_ref = db.collection("favorites").document(item["id"])
            _prev = _doc_ref.get()
            _prev_data = _prev.to_dict() if _prev.exists else {}
            _added_at = _prev_data.get("addedAt") or firestore.SERVER_TIMESTAMP
            payload = {
                "id": item["id"],
                "title": item["title"],
                "year": item.get("year"),
                "imdb": imdb_id,
                "poster": item.get("poster"),
                "imdbRating": imdb_rating,
                "rt": rt_score,
                "cineselectRating": cs_score,
                "type": media_key,
                "addedAt": _added_at,
            }
            _doc_ref.set(payload)
            append_seed_rating(
                imdb_id=imdb_id,
                title=item["title"],
                year=item.get("year"),
                imdb_rating=imdb_rating,
                rt_score=rt_score,
            )

        if selections and st.button("➕ Add Selected to Favorites"):
            for item in selections:
                add_to_favorites(item, cs_score=101)
            st.success(f"{len(selections)} öğe favorilere eklendi.")
            _safe_set_state("query_input", "")
            _safe_set_state("query", "")
            _safe_set_state("clear_search", True)
            st.rerun()

st.divider()
st.subheader("🎬 Film / Dizi Listesi")

if "fav_section" not in st.session_state:
    _safe_set_state("fav_section", "📌 İzlenecekler")
# Custom radio with rerun on change
new_section = st.radio(
    "Liste türü:",
    ["📌 İzlenecekler", "🎬 İzlenenler", "🖤 Blacklist"],
    index=0,
    horizontal=True,
    key="fav_section_radio"
)
if new_section != st.session_state.get("fav_section", ""):
    st.session_state["fav_section"] = new_section
    st.rerun()
fav_section = st.session_state["fav_section"]

sort_option = st.selectbox(
    "Sort by:", ["IMDb", "RT", "CineSelect", "Year"], index=2, key="fav_sort"
)


def show_favorites(fav_type, label, favorites=None):
    # Her zaman Firestore'dan oku; sadece Firestore'dan gelen veriyi kullan
    firestore_favorites = [
        doc.to_dict() for doc in db.collection("favorites").where("type", "==", fav_type).stream()
        if doc.to_dict().get("status") in ("to_watch", None, "")
    ]
    favorites = firestore_favorites
    favorites = sorted(favorites, key=get_sort_key, reverse=True)

    # --- Incremental scroll for Izlenecekler ---
    page_size = 50
    page_key = f"{st.session_state['fav_section']}_{fav_type}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    end_idx = st.session_state[page_key] * page_size
    display_favorites = favorites[:end_idx]

    st.markdown(f"### 📁 {label}")
    for idx, fav in enumerate(display_favorites):
        # Güvenli kimlik: id, imdbID, tmdb_id, key
        fid = fav.get("id") or fav.get("imdbID") or fav.get("tmdb_id") or fav.get("key")
        if not fid:
            fid = f"unknown_{idx}"
        imdb_display = (
            f"{float(fav.get('imdbRating', 0) or 0):.1f}"
            if fav.get('imdbRating') not in (None, "", "N/A") and isinstance(fav.get('imdbRating', 0), (int, float))
            else "N/A"
        )
        _rt_val = fav.get('rt', None)
        try:
            _rt_val_num = int(float(_rt_val)) if _rt_val not in (None, "", "N/A") else 0
        except Exception:
            _rt_val_num = 0
        rt_display = f"{_rt_val_num}%" if _rt_val_num > 0 else "N/A"
        cols = st.columns([1, 5, 1])
        with cols[0]:
            if show_posters and fav.get("poster"):
                poster_url = fav.get("Poster") or fav.get("poster_path") or fav.get("poster")
                imdb_id = fav.get("imdbID") or fav.get("imdb_id") or fav.get("imdb") or ""
                if imdb_id:
                    imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                else:
                    imdb_url = fav.get("imdb_url", "")
                if poster_url:
                    if imdb_url:
                        st.markdown(
                            f"<a href='{imdb_url}' target='_blank'>"
                            f"<img src='{poster_url}' alt='{fav.get('Title', fav.get('title', ''))}' width='120'/>"
                            "</a>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.image(poster_url, width=120)
        with cols[1]:
            st.markdown(f"**{idx+1}. {fav['title']} ({fav['year']})** | ⭐ IMDb: {imdb_display} | 🍅 RT: {rt_display} | 🎯 CS: {fav.get('cineselectRating', 'N/A')}")
            # --- Comments section: İzlenenler-style logic, with sort, edit, delete, and add ---
            comments = fav.get("comments", [])
            from datetime import datetime as _dt
            comments_sorted = sorted(comments, key=lambda c: parse_turkish_or_iso_date(c.get("date")), reverse=True)
            for c_idx, c in enumerate(comments_sorted):
                text = c.get("text", "")
                who = c.get("watchedBy", "")
                date = c.get("date", "")
                comment_row_cols = st.columns([8, 1, 1])
                with comment_row_cols[0]:
                    st.write(f"💬 {text} — ({who}) • {date}")
                with comment_row_cols[1]:
                    edit_mode_key = f"to_watch_comment_edit_mode_{fid}_{c_idx}"
                    if st.button("✏️", key=f"to_watch_comment_edit_{fid}_{c_idx}"):
                        _safe_set_state(edit_mode_key, True)
                        st.rerun()
                with comment_row_cols[2]:
                    if st.button("🗑️", key=f"to_watch_comment_del_{fid}_{c_idx}"):
                        new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                        db.collection("favorites").document(fid).update({"comments": new_comments})
                        fav["comments"] = new_comments
                        # update session_state immediately after Firestore update (mirror İzlenenler)
                        for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                            if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                item["comments"] = new_comments
                                break
                        st.success("🗑️ Yorum silindi!")
                        st.rerun()
                # Inline edit UI if in edit mode
                if st.session_state.get(edit_mode_key, False):
                    edit_text_key = f"to_watch_comment_edit_text_{fid}_{c_idx}"
                    edit_who_key = f"to_watch_comment_edit_who_{fid}_{c_idx}"
                    if edit_text_key not in st.session_state:
                        _safe_set_state(edit_text_key, text)
                    default_who = (who or "ss")
                    edit_cols = st.columns([3, 2])
                    with edit_cols[0]:
                        new_text = st.text_area(
                            "Yorumu düzenle",
                            key=edit_text_key,
                            height=80,
                            label_visibility="collapsed",
                        )
                    with edit_cols[1]:
                        new_who = st.selectbox(
                            "Yorumu kim yaptı?",
                            ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"],
                            index=(["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"].index(default_who) if default_who in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"] else 1),
                            key=edit_who_key
                        )
                    save_col, cancel_col = st.columns([1, 1])
                    with save_col:
                        if st.button("💾 Kaydet", key=f"to_watch_comment_save_{fid}_{c_idx}"):
                            now_str = format_turkish_datetime(_dt.now())
                            comments_sorted[c_idx] = {
                                "text": new_text.strip(),
                                "watchedBy": new_who,
                                "date": now_str
                            }
                            db.collection("favorites").document(fid).update({"comments": comments_sorted})
                            fav["comments"] = comments_sorted
                            # update session_state immediately after Firestore update (mirror İzlenenler)
                            for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                                if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                    item["comments"] = comments_sorted
                                    break
                            st.success("✏️ Yorum güncellendi!")
                            _safe_set_state(edit_mode_key, False)
                            st.rerun()
                    with cancel_col:
                        if st.button("❌ İptal", key=f"to_watch_comment_cancel_{fid}_{c_idx}"):
                            _safe_set_state(edit_mode_key, False)
                            st.rerun()
            # --- Yorum Ekle expander, İzlenenler-style, immediately after comments ---
            with st.expander("💬 Yorum Ekle"):
                # st.write("DEBUG fav keys:", list(fav.keys()))
                # st.write("DEBUG fav data:", fav)
                comment_key = f"to_watch_comment_add_{fid}"
                comment_wb_key = f"to_watch_comment_add_wb_{fid}"
                if comment_key not in st.session_state:
                    _safe_set_state(comment_key, "")
                comment_text = st.text_area(
                    "Yorum ekle",
                    key=comment_key,
                    height=100,
                    label_visibility="visible"
                )
                comment_wb_val = st.selectbox(
                    "Yorumu kim yaptı?",
                    ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"],
                    index=(["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"].index(st.session_state.get(comment_wb_key, "ss"))
                           if st.session_state.get(comment_wb_key, "ss") in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"] else 1),
                    key=comment_wb_key,
                    label_visibility="visible"
                )
                comment_btn_key = f"to_watch_comment_add_btn_{fid}"
                comments = fav.get("comments", [])
                new_comments = list(comments) if comments else []
                if st.button("💬 Comment yap", key=comment_btn_key):
                    now_str = format_turkish_datetime(_dt.now())
                    comment_full = comment_text.strip()
                    who_val = st.session_state.get(comment_wb_key, "")
                    if comment_full and who_val:
                        new_comment = {
                            "text": comment_full,
                            "watchedBy": who_val,
                            "date": now_str,
                        }
                        new_comments.append(new_comment)
                        # 1. Firestore update
                        db.collection("favorites").document(fid).update({"comments": new_comments})
                        # 2. Update fav["comments"]
                        fav["comments"] = new_comments
                        # 3. session_state güncellemesi
                        for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie"
                                     else st.session_state["favorite_series"]):
                            if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                item["comments"] = new_comments
                                break
                        # 4. _safe_set_state(comment_key, "")
                        _safe_set_state(comment_key, "")
                        # 5. _safe_set_state(comment_wb_key, "ss")
                        _safe_set_state(comment_wb_key, "ss")
                        # 6. st.success
                        st.success("💬 Yorum kaydedildi!")
                        # 7. st.rerun()
                        st.rerun()
        with cols[2]:
            with st.expander("✨ Options"):
                # --- (Comment edit/delete UI is now inline under the movie details, not in Options expander) ---
                # --- Status selectbox (short labels) and all action buttons grouped in expander ---
                status_options = ["to_watch", "öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g", "n/w", "🖤 BL"]
                # Compute current status string with new logic
                if fav.get("status") == "to_watch":
                    current_status_str = "to_watch"
                elif fav.get("status") == "watched":
                    wb = fav.get("watchedBy")
                    if wb in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"]:
                        current_status_str = wb
                    else:
                        current_status_str = "n/w"
                elif fav.get("status") == "blacklist":
                    current_status_str = "🖤 BL"
                else:
                    current_status_str = "to_watch"
                status_select = st.selectbox(
                    "Watched by",
                    status_options,
                    index=status_options.index(current_status_str) if current_status_str in status_options else 0,
                    key=f"status_{fid}"
                )
                from datetime import datetime
                # --- Hızlı geçiş mantığı: İzlenecekler'de statü değişikliği anında diğer listeye aktar ---
                # Sadece "to_watch" listesindeyken hızlı geçiş uygula, onay ve yorum isteme
                if status_select != current_status_str:
                    doc_ref = db.collection("favorites").document(fid)
                    if status_select == "to_watch":
                        doc_ref.update({
                            "status": "to_watch",
                            "watchedBy": None,
                            "watchedAt": None,
                            "watchedEmoji": None,
                            "blacklistedBy": None,
                            "blacklistedAt": None,
                        })
                        # Update session_state
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                item.update({
                                    "status": "to_watch",
                                    "watchedBy": None,
                                    "watchedAt": None,
                                    "watchedEmoji": None,
                                    "blacklistedBy": None,
                                    "blacklistedAt": None,
                                })
                                break
                        st.session_state["fav_section"] = "📌 İzlenecekler"
                        st.success(f"✅ {fav['title']} durumu güncellendi: to_watch")
                        st.rerun()
                    elif status_select in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g", "n/w"]:
                        now_str = format_turkish_datetime(datetime.now())
                        doc_ref.update({
                            "status": "watched",
                            "watchedBy": None if status_select == "n/w" else status_select,
                            "watchedAt": now_str,
                            "cineselectRating": 60 if status_select == "n/w" else fav.get("cineselectRating", 60),
                            "watchedEmoji": "😐" if status_select == "n/w" else fav.get("watchedEmoji", "😐"),
                            "blacklistedBy": None,
                            "blacklistedAt": None,
                        })
                        # Update session_state
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                item.update({
                                    "status": "watched",
                                    "watchedBy": None if status_select == "n/w" else status_select,
                                    "watchedAt": now_str,
                                    "cineselectRating": 60 if status_select == "n/w" else fav.get("cineselectRating", 60),
                                    "watchedEmoji": "😐" if status_select == "n/w" else fav.get("watchedEmoji", "😐"),
                                    "blacklistedBy": None,
                                    "blacklistedAt": None,
                                })
                                break
                        st.session_state["fav_section"] = "🎬 İzlenenler"
                        st.success(f"✅ {fav['title']} durumu güncellendi: watched ({status_select})")
                        st.rerun()
                    elif status_select == "🖤 BL":
                        now_str = format_turkish_datetime(datetime.now())
                        doc_ref.update({
                            "status": "blacklist",
                            "blacklistedBy": "🖤 BL",
                            "blacklistedAt": now_str,
                            "watchedBy": None,
                            "watchedAt": None,
                        })
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                item.update({
                                    "status": "blacklist",
                                    "blacklistedBy": "🖤 BL",
                                    "blacklistedAt": now_str,
                                    "watchedBy": None,
                                    "watchedAt": None,
                                })
                                break
                        st.session_state["fav_section"] = "🖤 Blacklist"
                        st.success(f"✅ {fav['title']} blacklist'e taşındı!")
                        st.rerun()
                # --- Action buttons: edit, pin, etc. ---
                if st.button("🔄 IMDb&RT", key=f"refresh_{fid}"):
                    imdb_id = fav.get("imdb")
                    # Eğer imdb_id boşsa TMDb'den al
                    if not imdb_id:
                        imdb_id = get_imdb_id_from_tmdb(fav.get("title"), fav.get("year"), is_series=(fav.get("type")=="show"))
                        st.info(f"🎬 IMDb ID TMDb'den alındı: {imdb_id}")
                    if imdb_id:
                        stats = get_ratings(imdb_id)
                        st.write("🔍 get_ratings output:", stats)
                        imdb_rating = stats.get("imdb_rating") if stats else None
                        rt_score = stats.get("rt") if stats else None
                        db.collection("favorites").document(fid).update({
                            "imdb": imdb_id,
                            "imdbRating": float(imdb_rating) if imdb_rating is not None else 0.0,
                            "rt": int(rt_score) if rt_score is not None else 0,
                        })
                        st.success(f"✅ IMDb/RT güncellendi: {fav.get('title','?')} (IMDb={imdb_rating}, RT={rt_score})")
                        st.rerun()
                    else:
                        st.error(f"❌ IMDb ID bulunamadı: {fav.get('title')}")
                if st.button("✏️", key=f"edit_{fid}"):
                    _safe_set_state(f"edit_mode_{fid}", True)
                # PIN FIRST: handle "Başa tuttur" BEFORE rendering input so it reflects new value immediately
                pin_now = st.button("📌 Başa tuttur", key=f"pin_{fid}")
                if pin_now:
                    # Find the maximum visible CS among current items and set to min(100, max_val + 1)
                    try:
                        visible_cs = [
                            int(x.get("cineselectRating") or 0)
                            for x in favorites
                            if isinstance(x.get("cineselectRating"), (int, float)) and int(x.get("cineselectRating") or 0) > 0
                        ]
                    except Exception:
                        visible_cs = []
                    if visible_cs:
                        base = max(visible_cs)
                    else:
                        base = 50
                    pin_val = min(100, base + 1)
                    # Update Firestore document immediately
                    db.collection("favorites").document(fid).update({"cineselectRating": pin_val})
                    # Update session_state as well
                    for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                        if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                            item["cineselectRating"] = pin_val
                            break
                    _safe_set_state(f"input_{fid}", pin_val)
                    st.success(f"📌 CineSelect puanı {pin_val} olarak güncellendi ve başa taşındı!")
                    st.rerun()
                if st.session_state.get(f"edit_mode_{fid}", False):
                    i_key = f"input_{fid}"
                    current = _clamp_cs(fav.get("cineselectRating", 50))
                    st.number_input(
                        "🎯 CS:",
                        min_value=1,
                        max_value=150,
                        value=st.session_state.get(i_key, current),
                        step=1,
                        key=i_key
                    )
                    cols_edit = st.columns([1,2])
                    with cols_edit[0]:
                        if st.button("✅ Kaydet", key=f"save_{fid}"):
                            new_val = _clamp_cs(st.session_state.get(i_key, current))
                            db.collection("favorites").document(fid).update({"cineselectRating": new_val})
                            # Update session_state as well
                            for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                                if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                    item["cineselectRating"] = new_val
                                    break
                            st.success(f"✅ {fav['title']} güncellendi (CS={new_val}).")
                            _safe_set_state(f"edit_mode_{fid}", False)
                            st.rerun()
                    with cols_edit[1]:
                        st.caption("🔧 İpucu: 'Başa tuttur' butonuna bastıktan sonra 'Kaydet' ile kalıcılaştır.")

    # Eğer daha fazla öğe varsa buton ekle (after rendering all currently visible items)
    if end_idx < len(favorites):
        if st.button("🔽 Daha Fazla Yükle", key=f"load_more_{fav_type}_{page_key}_{end_idx}"):
            st.session_state[page_key] += 1
            st.rerun()

if fav_section == "📌 İzlenecekler":
    # Improved favorite counts display (before showing lists)
    def show_favorites_count():
        movies = st.session_state.get("favorite_movies", [])
        series = st.session_state.get("favorite_series", [])
        movie_count = len([m for m in movies if (m.get("type") == "movie" or m.get("type") is None) and m.get("status") == "to_watch"])
        # For series: count items with status "to_watch" or status is None
        series_count = len([s for s in series if (s.get("type") == "show") and (s.get("status") == "to_watch" or s.get("status") is None)])
        total = movie_count + series_count
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Filmler", movie_count)
        with col2:
            st.metric("Diziler", series_count)
        with col3:
            st.metric("Toplam", total)
        # Alternatively, for a single centered bold line:
        # st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:20px;'>Filmler: {movie_count} &nbsp;|&nbsp; Diziler: {series_count} &nbsp;|&nbsp; Toplam: {total}</div>", unsafe_allow_html=True)
    show_favorites_count()
    # --- Move sync order radio below "Sort by" selectbox ---
    # The selectbox for sort_option is already defined above; insert the radio after it.
    sync_order_to_watch = st.radio(
        "Sync sıralaması (İzlenecekler):",
        ["CineSelect", "IMDb", "Year"],
        index=0,
        key="sync_order_to_watch",
        horizontal=True
    )
    if media_type == "Movie":
        show_favorites("movie", "Filmler")
    elif media_type == "TV Show":
        show_favorites("show", "Diziler")
elif fav_section == "🎬 İzlenenler":
    st.markdown("---")
    # Insert sort option selectbox for watched items
    sort_option_watched = st.selectbox(
        "Sort by (Watched):",
        ["Watched Date", "IMDb", "RT", "CineSelect", "Year"],
        index=0,
        key="watched_sort"
    )
    # --- Move sync order radio below watched sort selectbox ---
    sync_order_watched = st.radio(
        "Sync sıralaması (İzlenenler):",
        ["Watched Date", "CineSelect", "IMDb", "Year"],
        index=0,
        key="sync_order_watched",
        horizontal=True
    )
    st.subheader(f"🎬 İzlenenler (sort: {sort_option_watched})")
    watched_docs = db.collection("favorites").where("status", "==", "watched").stream()
    watched_items = [doc.to_dict() for doc in watched_docs]
    def _watched_sort_key(fav):
        return parse_turkish_or_iso_date(fav.get("watchedAt"))
    # Conditional sorting based on selected sort option
    if sort_option_watched == "Watched Date":
        watched_items = sorted(watched_items, key=_watched_sort_key, reverse=True)
    elif sort_option_watched == "IMDb":
        watched_items = sorted(watched_items, key=lambda f: float(f.get("imdbRating") or 0), reverse=True)
    elif sort_option_watched == "RT":
        watched_items = sorted(watched_items, key=lambda f: float(f.get("rt") or 0), reverse=True)
    elif sort_option_watched == "CineSelect":
        watched_items = sorted(watched_items, key=lambda f: int(f.get("cineselectRating") or 0), reverse=True)
    elif sort_option_watched == "Year":
        watched_items = sorted(watched_items, key=lambda f: int(f.get("year") or 0), reverse=True)
    from datetime import datetime as _dt
    from collections import defaultdict, OrderedDict
    import calendar
    import locale
    # Turkish short month names
    TURKISH_MONTHS = {
        1: "Oca", 2: "Şub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz",
        7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki", 11: "Kas", 12: "Ara"
    }
    # Group watched_items by (year, month)
    groups = defaultdict(list)
    for fav in watched_items:
        dt = parse_turkish_or_iso_date(fav.get("watchedAt"))
        if not dt or not hasattr(dt, "year"):
            # fallback: group as (None, None)
            key = (None, None)
        else:
            key = (dt.year, dt.month)
        groups[key].append(fav)
    # Sort groups by (year, month) descending
    group_keys_sorted = sorted(
        [k for k in groups.keys() if k != (None, None)],
        key=lambda x: (x[0], x[1]),
        reverse=True
    )
    # Add (None, None) at the end if exists
    if (None, None) in groups:
        group_keys_sorted.append((None, None))
    # For the most recent 2 months, show directly; others collapsed
    for group_idx, group_key in enumerate(group_keys_sorted):
        items = groups[group_key]
        # Format header
        year, month = group_key
        if year is not None and month is not None:
            month_label = TURKISH_MONTHS.get(month, f"{month:02d}")
            year_label = f"'{str(year)[2:]}"
            group_label = f"{month_label} {year_label}"
        else:
            group_label = "Diğer"
        # Show header
        if group_idx < 2:
            st.markdown(f"#### {group_label}")
            for idx, fav in enumerate(items, start=1):
                imdb_display = f"{float(fav.get('imdbRating', 0) or 0):.1f}" if fav.get('imdbRating') else "N/A"
                rt_val = fav.get("rt", 0)
                try:
                    rt_num = int(float(rt_val)) if rt_val not in (None, "", "N/A") else 0
                except Exception:
                    rt_num = 0
                rt_display = f"{rt_num}%" if rt_num > 0 else "N/A"
                cols = st.columns([1, 5, 1])
                with cols[0]:
                    poster_url = fav.get("Poster") or fav.get("poster_path") or fav.get("poster")
                    imdb_id = fav.get("imdbID") or fav.get("imdb_id") or fav.get("imdb") or ""
                    if imdb_id:
                        imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                    else:
                        imdb_url = fav.get("imdb_url", "")
                    if poster_url:
                        if imdb_url:
                            st.markdown(
                                f"<a href='{imdb_url}' target='_blank'>"
                                f"<img src='{poster_url}' alt='{fav.get('Title', fav.get('title', ''))}' width='120'/>"
                                "</a>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.image(poster_url, width=120)
                with cols[1]:
                    emoji = fav.get("watchedEmoji") or "😐"
                    title_str = f"**{idx}. {fav.get('title')} ({fav.get('year')})**"
                    if emoji:
                        title_str += f" {emoji}"
                    st.markdown(
                        f"{title_str} | ⭐ IMDb: {imdb_display} | 🍅 RT: {rt_display} | 🎯 CS: {fav.get('cineselectRating','N/A')} | 👤 {fav.get('watchedBy','?')} | ⏰ {fav.get('watchedAt','?')}",
                        unsafe_allow_html=True
                    )
                    # --- Comments Section: inline, with edit/delete and inline edit UI (like Blacklist/To-Watch) ---
                    comments = fav.get("comments", [])
                    comments_sorted = sorted(comments, key=lambda c: parse_turkish_or_iso_date(c.get("date")), reverse=True)
                    for c_idx, c in enumerate(comments_sorted):
                        text = c.get("text", "")
                        who = c.get("watchedBy", "")
                        date = c.get("date", "")
                        comment_row_cols = st.columns([8, 1, 1])
                        with comment_row_cols[0]:
                            st.write(f"💬 {text} — ({who}) • {date}")
                        with comment_row_cols[1]:
                            edit_mode_key = f"watched_comment_edit_mode_{fav['id']}_{c_idx}"
                            if st.button("✏️", key=f"watched_comment_edit_{fav['id']}_{c_idx}"):
                                _safe_set_state(edit_mode_key, True)
                                st.rerun()
                        with comment_row_cols[2]:
                            if st.button("🗑️", key=f"watched_comment_del_{fav['id']}_{c_idx}"):
                                new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                                db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                                fav["comments"] = new_comments
                                for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                                    if item.get("id") == fav["id"]:
                                        item["comments"] = new_comments
                                        break
                                st.success("🗑️ Yorum silindi!")
                                st.rerun()
                        if st.session_state.get(edit_mode_key, False):
                            edit_text_key = f"watched_comment_edit_text_{fav['id']}_{c_idx}"
                            edit_who_key = f"watched_comment_edit_who_{fav['id']}_{c_idx}"
                            if edit_text_key not in st.session_state:
                                _safe_set_state(edit_text_key, text)
                            default_who = (who or "ss")
                            edit_cols = st.columns([3, 2])
                            with edit_cols[0]:
                                new_text = st.text_area(
                                    "Yorumu düzenle",
                                    key=edit_text_key,
                                    height=80,
                                    label_visibility="collapsed",
                                )
                            with edit_cols[1]:
                                new_who = st.selectbox(
                                    "Yorumu kim yaptı?",
                                    ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"],
                                    index=( ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"].index(default_who) if default_who in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"] else 1 ),
                                    key=edit_who_key
                                )
                            save_col, cancel_col = st.columns([1, 1])
                            with save_col:
                                if st.button("💾 Kaydet", key=f"watched_comment_save_{fav['id']}_{c_idx}"):
                                    now_str = format_turkish_datetime(_dt.now())
                                    comments_sorted[c_idx] = {
                                        "text": new_text.strip(),
                                        "watchedBy": new_who,
                                        "date": now_str
                                    }
                                    db.collection("favorites").document(fav["id"]).update({"comments": comments_sorted})
                                    fav["comments"] = comments_sorted
                                    for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                                        if item.get("id") == fav["id"]:
                                            item["comments"] = comments_sorted
                                            break
                                    st.success("✏️ Yorum güncellendi!")
                                    _safe_set_state(edit_mode_key, False)
                                    st.rerun()
                            with cancel_col:
                                if st.button("❌ İptal", key=f"watched_comment_cancel_{fav['id']}_{c_idx}"):
                                    _safe_set_state(edit_mode_key, False)
                                    st.rerun()
                    with st.expander("💬 Yorum Ekle"):
                        comment_key = f"watched_comment_add_{fav['id']}"
                        comment_wb_key = f"watched_comment_add_wb_{fav['id']}"
                        if comment_key not in st.session_state:
                            _safe_set_state(comment_key, "")
                        comment_text = st.text_area(
                            "Yorum ekle",
                            key=comment_key,
                            height=100,
                            label_visibility="visible"
                        )
                        comment_wb_val = st.selectbox(
                            "Yorumu kim yaptı?",
                            ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"],
                            index=(["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"].index(st.session_state.get(comment_wb_key, "ss"))
                                   if st.session_state.get(comment_wb_key, "ss") in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"] else 1),
                            key=comment_wb_key,
                            label_visibility="visible"
                        )
                        comment_btn_key = f"watched_comment_add_btn_{fav['id']}"
                        comments = fav.get("comments", [])
                        new_comments = list(comments) if comments else []
                        if st.button("💬 Comment yap", key=comment_btn_key):
                            now_str = format_turkish_datetime(_dt.now())
                            comment_full = comment_text.strip()
                            who_val = st.session_state.get(comment_wb_key, "")
                            if comment_full and who_val:
                                new_comment = {
                                    "text": comment_full,
                                    "watchedBy": who_val,
                                    "date": now_str,
                                }
                                new_comments.append(new_comment)
                                db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                                fav["comments"] = new_comments
                                for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie"
                                             else st.session_state["favorite_series"]):
                                    if item.get("id") == fav["id"]:
                                        item["comments"] = new_comments
                                        break
                                _safe_set_state(comment_key, "")
                                _safe_set_state(comment_wb_key, "ss")
                                st.success("💬 Yorum kaydedildi!")
                                st.rerun()
                # ---- Inserted/Updated: Options expander for watched items ----
                with cols[2]:
                    with st.expander("✨ Options"):
                        # --- Status selectbox and fast transitions for WATCHED section ---
                        status_options = ["to_watch", "öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g", "n/w", "🖤 BL"]

                        # Compute current status string (same mapping used in To-Watch list)
                        if fav.get("status") == "to_watch":
                            current_status_str = "to_watch"
                        elif fav.get("status") == "blacklist":
                            current_status_str = "🖤 BL"
                        elif fav.get("status") == "watched":
                            wb = fav.get("watchedBy")
                            if wb in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g"]:
                                current_status_str = wb
                            else:
                                current_status_str = "n/w"
                        else:
                            current_status_str = "to_watch"

                        status_select = st.selectbox(
                            "Watched by",
                            status_options,
                            index=status_options.index(current_status_str) if current_status_str in status_options else 0,
                            key=f"watched_status_{fav['id']}"
                        )

                        from datetime import datetime
                        fid = fav.get("id")
                        fav_type = (fav.get("type") or "movie")

                        if status_select != current_status_str:
                            doc_ref = db.collection("favorites").document(fid)

                            if status_select == "to_watch":
                                # Move back to To-Watch quickly
                                doc_ref.update({
                                    "status": "to_watch",
                                    "watchedBy": None,
                                    "watchedAt": None,
                                    "watchedEmoji": None,
                                    "blacklistedBy": None,
                                    "blacklistedAt": None,
                                })
                                # mirror session_state
                                for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                                    if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                        item.update({
                                            "status": "to_watch",
                                            "watchedBy": None,
                                            "watchedAt": None,
                                            "watchedEmoji": None,
                                            "blacklistedBy": None,
                                            "blacklistedAt": None,
                                        })
                                        break
                                st.session_state["fav_section"] = "📌 İzlenecekler"
                                st.success(f"✅ {fav['title']} durumu güncellendi: to_watch")
                                st.rerun()

                            elif status_select in ["öz", "ss", "öz❤️ss", "ds", "gs", "s❤️d", "s❤️g", "n/w"]:
                                # Stay in Watched with quick update
                                now_str = format_turkish_datetime(datetime.now())
                                doc_ref.update({
                                    "status": "watched",
                                    "watchedBy": None if status_select == "n/w" else status_select,
                                    "watchedAt": now_str,
                                    "cineselectRating": 60 if status_select == "n/w" else fav.get("cineselectRating", 60),
                                    "watchedEmoji": "😐" if status_select == "n/w" else fav.get("watchedEmoji", "😐"),
                                    "blacklistedBy": None,
                                    "blacklistedAt": None,
                                })
                                for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                                    if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                        item.update({
                                            "status": "watched",
                                            "watchedBy": None if status_select == "n/w" else status_select,
                                            "watchedAt": now_str,
                                            "cineselectRating": 60 if status_select == "n/w" else fav.get("cineselectRating", 60),
                                            "watchedEmoji": "😐" if status_select == "n/w" else fav.get("watchedEmoji", "😐"),
                                            "blacklistedBy": None,
                                            "blacklistedAt": None,
                                        })
                                        break
                                st.session_state["fav_section"] = "🎬 İzlenenler"
                                st.success(f"✅ {fav['title']} durumu güncellendi: watched ({status_select})")
                                st.rerun()

                            elif status_select == "🖤 BL":
                                # --- FAST SWITCH to Blacklist (requested fix) ---
                                now_str = format_turkish_datetime(datetime.now())
                                doc_ref.update({
                                    "status": "blacklist",
                                    "blacklistedBy": "🖤 BL",
                                    "blacklistedAt": now_str,
                                    "watchedBy": None,
                                    "watchedAt": None,
                                })
                                # mirror session_state
                                for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                                    if (item.get("id") or item.get("imdbID") or item.get("tmdb_id") or item.get("key")) == fid:
                                        item.update({
                                            "status": "blacklist",
                                            "blacklistedBy": "🖤 BL",
                                            "blacklistedAt": now_str,
                                            "watchedBy": None,
                                            "watchedAt": None,
                                        })
                                        break
                                # Switch UI section immediately like To-Watch fast path
                                st.session_state["fav_section"] = "🖤 Blacklist"
                                st.success(f"✅ {fav['title']} blacklist'e taşındı!")
                                st.rerun()
