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

# --- Cached Firestore favorites loader ---
@st.cache_data(ttl=240)  # Cache Firestore favorites for 240 seconds
def load_favorites():
    db = get_firestore()
    # Fetch with IN queries to include legacy values
    movie_docs = db.collection("favorites").where("type", "in", ["movie", "film"]).stream()
    show_docs  = db.collection("favorites").where("type", "in", ["show", "series", "tv", "tvshow"]).stream()

    movies = [doc.to_dict() for doc in movie_docs]
    shows  = [doc.to_dict() for doc in show_docs]

    # Normalize type for all items
    for item in movies:
        t = (item.get("type") or "").lower()
        if t in ["tv", "tvshow", "show", "series"]:
            item["type"] = "show"
        else:
            item["type"] = "movie"
    for item in shows:
        t = (item.get("type") or "").lower()
        if t in ["tv", "tvshow", "show", "series"]:
            item["type"] = "show"
        else:
            # Correction: shows collection should always be "show"
            item["type"] = "show"
    # Ensure both lists are in session_state correctly
    st.session_state["favorite_movies"] = movies
    st.session_state["favorite_series"] = shows
    return movies, shows
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
movies, shows = load_favorites()
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

# --- Quick Toolbar (always visible) ---
# Ensure required defaults
if "show_posters" not in st.session_state:
    st.session_state["show_posters"] = True
if "sync_sort_mode" not in st.session_state:
    st.session_state["sync_sort_mode"] = "cc"

tcol1, tcol2, tcol3, tcol_sp = st.columns([1.2, 1.6, 1.6, 4])
with tcol1:
    if st.button("🖼️ Toggle Posters", key="toolbar_toggle_posters"):
        st.session_state["show_posters"] = not st.session_state.get("show_posters", True)

with tcol2:
    if st.button("📂 JSON & CSV Sync", key="toolbar_sync"):
        sync_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "cc"))
        st.success("✅ favorites_stw.json ve seed_ratings.csv senkronize edildi.")

with tcol3:
    if st.button("📊 Favori Sayıları", key="toolbar_counts"):
        show_favorites_count()
# --- /Quick Toolbar ---


# --- Options Expander (All action buttons grouped) ---
def show_favorites_count():
    movie_docs = db.collection("favorites").where("type", "==", "movie").stream()
    series_docs = db.collection("favorites").where("type", "==", "show").stream()
    movie_count = len(list(movie_docs))
    series_count = len(list(series_docs))
    st.info(f"🎬 Favorite Movies: {movie_count} | 📺 Favorite TV Shows: {series_count}")

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
media_type = st.radio("Search type:", ["Movie", "TV Show", "Actor/Actress"], horizontal=True)

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
        if sort_name == "CineSelect":
            # For CineSelect, sort by CS descending as int
            return int(fav.get("cineselectRating") or 0)
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
    else:
        results = search_by_actor(query)

    try:
        results = sorted(results, key=lambda x: x.get("cineselectRating", 0), reverse=True)
    except:
        pass

    if not results:
        st.error("❌ No results found.")
    else:
        for idx, item in enumerate(results):
            st.divider()
            if item.get("poster") and show_posters:
                # Prefer an actual IMDb ID (e.g., "tt0133093"); fall back across common key variants
                imdb_id_link = str(
                    item.get("imdb")
                    or item.get("imdb_id")
                    or item.get("imdbID")
                    or ""
                ).strip()
                poster_url = item["poster"]
                if imdb_id_link.startswith("tt"):
                    st.markdown(
                        f'<a href="https://www.imdb.com/title/{imdb_id_link}/" target="_blank" rel="noopener">'
                        f'<img src="{poster_url}" width="180"/></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(poster_url, width=180)

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

            manual_key = f"manual_{item['id']}"
            manual_val = st.number_input(
                "🎯 CineSelect Rating:",
                min_value=1,
                max_value=100,
                value=st.session_state.get(manual_key, 50),
                step=1,
                key=manual_key
            )

            if st.button("Add to Favorites", key=f"btn_{item['id']}"):
                media_key = "movie" if media_type == "Movie" else ("show" if media_type == "TV Show" else "movie")

                from omdb import get_ratings, fetch_ratings  # ÜSTE ekli olsun

                # 1) IMDb ID garanti altına al
                imdb_id = (item.get("imdb") or "").strip()
                if not imdb_id or imdb_id == "tt0000000":
                    imdb_id = get_imdb_id_from_tmdb(
                        title=item["title"],
                        year=item.get("year"),
                        is_series=(media_key == "show"),
                )

                # 2) IMDb/RT puanlarını getir (ÖNCE yerel CSV, yoksa OMDb-ID, o da yoksa Title/Year)
                stats = {}
                raw_id = {}
                raw_title = {}
                source = None

                # a) yerel CSV
                seed_hit = read_seed_rating(imdb_id)
                if seed_hit and (seed_hit.get("imdb_rating") or seed_hit.get("rt")):
                    stats = {"imdb_rating": seed_hit.get("imdb_rating"), "rt": seed_hit.get("rt")}
                    source = "CSV"

                # b) CSV yoksa/eksikse OMDb by ID
                if not source:
                    if imdb_id:
                        stats = get_ratings(imdb_id) or {}
                        raw_id = (stats.get("raw") or {})
                        source = "CSV/OMDb-ID" if raw_id else None  # get_ratings CSV'den dönerse raw boş kalabilir

                # c) hâlâ boşsa OMDb by Title/Year
                if not stats or ((stats.get("imdb_rating") in (None, 0, "N/A")) and (stats.get("rt") in (None, 0, "N/A"))):
                    ir, rt, raw_title = fetch_ratings(item["title"], item.get("year"))
                    stats = {"imdb_rating": ir, "rt": rt}
                    if not source:
                        source = "OMDb-title"

                imdb_rating = float(stats.get("imdb_rating") or 0.0)
                rt_score    = int(stats.get("rt") or 0)

                # 🔎 DEBUG: Kaynak ve ham yanıtlar
                st.write(f"🔍 Source: {source or '—'} | 🆔 IMDb ID: {imdb_id or '—'} | ⭐ IMDb: {imdb_rating} | 🍅 RT: {rt_score}")

                # Extra, user-visible diagnostics
                error_msg = None
                if isinstance(raw_id, dict):
                    error_msg = raw_id.get("Error")
                if not error_msg and isinstance(raw_title, dict):
                    error_msg = raw_title.get("Error")

                if error_msg:
                    st.error(f"OMDb error: {error_msg}. Check OMDB_API_KEY.", icon="🚨")
                elif source == "CSV":
                    st.info("Source: seed_ratings.csv (cached)", icon="📂")
                elif source == "CSV/OMDb-ID":
                    st.info(f"Source: OMDb by IMDb ID ({imdb_id})", icon="🔎")
                else:
                    st.info(f"Source: OMDb by Title/Year ({item['title']} {item.get('year')})", icon="🔎")

                if raw_id:
                    import json as _json
                    st.caption("OMDb by ID (raw JSON)")
                    st.code(_json.dumps(raw_id, ensure_ascii=False, indent=2))
                if raw_title:
                    import json as _json
                    st.caption("OMDb by title (raw JSON)")
                    st.code(_json.dumps(raw_title, ensure_ascii=False, indent=2))
                # 3) Firestore'a yaz (varsa addedAt'i koru, yoksa ilk kez ekleniyorsa server timestamp ver)
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
                    "imdbRating": imdb_rating,                 # ✅ eklendi
                    "rt": rt_score,                            # ✅ CSV/OMDb’den gelen kesin değer
                    "cineselectRating": manual_val,
                    "type": media_key,
                    "addedAt": _added_at,
                }
                _doc_ref.set(payload)

                if _prev.exists:
                    # Bilgilendirme: Bu TMDB id zaten listedeydi; alanlar güncellendi.
                    try:
                        _when = _prev_data.get("addedAt")
                        _when_txt = format_turkish_datetime(_when) if hasattr(_when, "strftime") else str(_when)
                    except Exception:
                        _when_txt = "—"
                    st.info(f"ℹ️ Bu öğe zaten listendeydi (ilk eklenme: {_when_txt}); bilgiler güncellendi.", icon="ℹ️")
                # 4) seed_ratings.csv'ye (yoksa) ekle
                append_seed_rating(
                    imdb_id=imdb_id,
                    title=item["title"],
                    year=item.get("year"),
                    imdb_rating=imdb_rating,
                    rt_score=rt_score,
                )
                st.success(f"✅ {item['title']} added to favorites!")
                # clear search on next run to avoid "modified after instantiation" error
                st.session_state.clear_search = True
                # Let the user see the diagnostics before refresh
                st.toast("Refreshing…", icon="🔄")
                time.sleep(1.2)
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


def show_favorites(fav_type, label):
    # Fetch from Firestore: only items with status == "to_watch"
    to_watch_docs = db.collection("favorites").where("status", "==", "to_watch").stream()
    favorites = [doc.to_dict() for doc in to_watch_docs if (doc.to_dict().get("type") == fav_type)]
    favorites = sorted(favorites, key=get_sort_key, reverse=True)

    # --- Incremental scroll for Izlenecekler ---
    page_size = 50
    page_key = f"to_watch_page_{fav_type}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    # Slice favorites according to current page
    display_favorites = favorites[:page_size * st.session_state[page_key]]

    # Render only sliced favorites
    favorites = display_favorites

    # Inject JS to detect scroll bottom and auto-increase page
    scroll_js = f"""
    <script>
    var bottomSent = false;
    window.onscroll = function() {{
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 10) {{
            if (!bottomSent) {{
                fetch('/_stcore/streamlit/command', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{'command': 'increment_page', 'key': '{page_key}'}})
                }});
                bottomSent = true;
            }}
        }} else {{
            bottomSent = false;
        }}
    }};
    </script>
    """
    st.markdown(scroll_js, unsafe_allow_html=True)

    # Handle page increment command
    if "_command" in st.session_state and st.session_state["_command"].get("command") == "increment_page" and st.session_state["_command"].get("key") == page_key:
        st.session_state[page_key] += 1
        del st.session_state["_command"]
        st.rerun()

    st.markdown(f"### 📁 {label}")
    for idx, fav in enumerate(favorites):
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
                imdb_id_link = str(
                    fav.get("imdb") or fav.get("imdb_id") or fav.get("imdbID") or ""
                ).strip()
                poster_url = fav["poster"]
                if imdb_id_link and imdb_id_link.startswith("tt"):
                    st.markdown(
                        f'<a href="https://www.imdb.com/title/{imdb_id_link}/" target="_blank" rel="noopener">'
                        f'<img src="{poster_url}" width="120"/></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(poster_url, width=120)
        with cols[1]:
            st.markdown(f"**{idx+1}. {fav['title']} ({fav['year']})** | ⭐ IMDb: {imdb_display} | 🍅 RT: {rt_display} | 🎯 CS: {fav.get('cineselectRating', 'N/A')}")
            # --- Comments section: İzlenecekler - match İzlenenler 6-step logic for add, edit, delete ---
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
                    edit_mode_key = f"to_watch_comment_edit_mode_{fav['id']}_{c_idx}"
                    if st.button("✏️", key=f"to_watch_comment_edit_{fav['id']}_{c_idx}"):
                        _safe_set_state(edit_mode_key, True)
                        st.rerun()
                with comment_row_cols[2]:
                    if st.button("🗑️", key=f"comment_del_{fav['id']}_{c_idx}"):
                        new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                        # 1. Firestore update
                        db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                        # 2. Update fav["comments"]
                        fav["comments"] = new_comments
                        # 3. Update session_state
                        fav_type_val = fav.get("type", "").lower()
                        if fav_type_val in ["movie", "film"]:
                            target_list = st.session_state["favorite_movies"]
                        else:
                            target_list = st.session_state["favorite_series"]
                        for item in target_list:
                            if item.get("id") == fav["id"]:
                                item["comments"] = new_comments
                                break
                        # 4. Reset input state
                        comment_key = f"to_watch_comment_add_{fav['id']}"
                        comment_wb_key = f"to_watch_comment_add_wb_{fav['id']}"
                        _safe_set_state(comment_key, "")
                        _safe_set_state(comment_wb_key, "ss")
                        # 5. Success
                        st.success("🗑️ Yorum silindi!")
                        # 6. Rerun
                        st.rerun()
                # Inline edit UI if in edit mode
                if st.session_state.get(edit_mode_key, False):
                    edit_text_key = f"to_watch_comment_edit_text_{fav['id']}_{c_idx}"
                    edit_who_key = f"to_watch_comment_edit_who_{fav['id']}_{c_idx}"
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
                            ["öz", "ss", "öz❤️ss"],
                            index=(["öz", "ss", "öz❤️ss"].index(default_who) if default_who in ["öz", "ss", "öz❤️ss"] else 1),
                            key=edit_who_key
                        )
                    save_col, cancel_col = st.columns([1, 1])
                    with save_col:
                        # REPLACE WITH 6-STEP BLOCK
                        fav_type_val = fav.get("type", "").lower()
                        if fav_type_val in ["movie", "film"]:
                            target_list = st.session_state["favorite_movies"]
                        else:
                            target_list = st.session_state["favorite_series"]
                        if st.button("💾 Kaydet", key=f"comment_save_{fav['id']}_{c_idx}"):
                            now_str = format_turkish_datetime(_dt.now())
                            comments_sorted[c_idx] = {
                                "text": new_text.strip(),
                                "watchedBy": new_who,
                                "date": now_str,
                            }
                            # 1. Firestore update
                            db.collection("favorites").document(fav["id"]).update({"comments": comments_sorted})
                            # 2. Update fav["comments"]
                            fav["comments"] = comments_sorted
                            # 3. Update session_state
                            for item in target_list:
                                if item.get("id") == fav["id"]:
                                    item["comments"] = comments_sorted
                                    break
                            # 4. Reset edit mode
                            _safe_set_state(edit_mode_key, False)
                            # 5. Success
                            st.success("✏️ Yorum güncellendi!")
                            # 6. Rerun
                            st.rerun()
                    with cancel_col:
                        if st.button("❌ İptal", key=f"to_watch_comment_cancel_{fav['id']}_{c_idx}"):
                            _safe_set_state(edit_mode_key, False)
                            st.rerun()
            # --- Yorum Ekle expander, İzlenenler-style, immediately after comments ---
            with st.expander("💬 Yorum Ekle"):
                comment_key = f"to_watch_comment_add_{fav['id']}"
                comment_wb_key = f"to_watch_comment_add_wb_{fav['id']}"
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
                    ["öz", "ss", "öz❤️ss"],
                    index=(["öz", "ss", "öz❤️ss"].index(st.session_state.get(comment_wb_key, "ss"))
                           if st.session_state.get(comment_wb_key, "ss") in ["öz", "ss", "öz❤️ss"] else 1),
                    key=comment_wb_key,
                    label_visibility="visible"
                )
                comment_btn_key = f"to_watch_comment_add_btn_{fav['id']}"
                comments_existing = fav.get("comments", [])
                new_comments = list(comments_existing) if comments_existing else []
                # Determine target_list before add
                fav_type_val = fav.get("type", "").lower()
                if fav_type_val in ["movie", "film"]:
                    target_list = st.session_state["favorite_movies"]
                else:
                    target_list = st.session_state["favorite_series"]
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
                        db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                        # 2. Update fav["comments"]
                        fav["comments"] = new_comments
                        # 3. Update session_state
                        for item in target_list:
                            if item.get("id") == fav["id"]:
                                item["comments"] = new_comments
                                break
                        # 4. Reset input states
                        _safe_set_state(comment_key, "")
                        _safe_set_state(comment_wb_key, "ss")
                        # 5. Success
                        st.success("💬 Yorum kaydedildi!")
                        # 6. Rerun
                        st.rerun()
        with cols[2]:
            with st.expander("✨ Options"):
                # --- (Comment edit/delete UI is now inline under the movie details, not in Options expander) ---
                # --- Status selectbox (short labels) and all action buttons grouped in expander ---
                status_options = ["to_watch", "öz", "ss", "öz❤️ss", "n/w", "🖤 BL"]
                # Compute current status string with new logic
                if fav.get("status") == "to_watch":
                    current_status_str = "to_watch"
                elif fav.get("status") == "watched":
                    wb = fav.get("watchedBy")
                    if wb in ["öz", "ss", "öz❤️ss"]:
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
                    key=f"status_{fav['id']}"
                )
                from datetime import datetime
                # --- Hızlı geçiş mantığı: İzlenecekler'de statü değişikliği anında diğer listeye aktar ---
                # Sadece "to_watch" listesindeyken hızlı geçiş uygula, onay ve yorum isteme
                if status_select != current_status_str:
                    doc_ref = db.collection("favorites").document(fav["id"])
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
                            if item.get("id") == fav["id"]:
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
                    elif status_select in ["öz", "ss", "öz❤️ss", "n/w"]:
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
                            if item.get("id") == fav["id"]:
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
                            if item.get("id") == fav["id"]:
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
                if st.button("✏️", key=f"edit_{fav['id']}"):
                    _safe_set_state(f"edit_mode_{fav['id']}", True)
                # PIN FIRST: handle "Başa tuttur" BEFORE rendering input so it reflects new value immediately
                pin_now = st.button("📌 Başa tuttur", key=f"pin_{fav['id']}")
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
                    db.collection("favorites").document(fav["id"]).update({"cineselectRating": pin_val})
                    # Update session_state as well
                    for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                        if item.get("id") == fav["id"]:
                            item["cineselectRating"] = pin_val
                            break
                    _safe_set_state(f"input_{fav['id']}", pin_val)
                    st.success(f"📌 CineSelect puanı {pin_val} olarak güncellendi ve başa taşındı!")
                    st.rerun()
                if st.session_state.get(f"edit_mode_{fav['id']}", False):
                    i_key = f"input_{fav['id']}"
                    current = _clamp_cs(fav.get("cineselectRating", 50))
                    st.number_input(
                        "🎯 CS:",
                        min_value=1,
                        max_value=100,
                        value=st.session_state.get(i_key, current),
                        step=1,
                        key=i_key
                    )
                    cols_edit = st.columns([1,2])
                    with cols_edit[0]:
                        if st.button("✅ Kaydet", key=f"save_{fav['id']}"):
                            new_val = _clamp_cs(st.session_state.get(i_key, current))
                            db.collection("favorites").document(fav["id"]).update({"cineselectRating": new_val})
                            # Update session_state as well
                            for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                                if item.get("id") == fav["id"]:
                                    item["cineselectRating"] = new_val
                                    break
                            st.success(f"✅ {fav['title']} güncellendi (CS={new_val}).")
                            _safe_set_state(f"edit_mode_{fav['id']}", False)
                            st.rerun()
                    with cols_edit[1]:
                        st.caption("🔧 İpucu: 'Başa tuttur' butonuna bastıktan sonra 'Kaydet' ile kalıcılaştır.")


# --- Move render_favorite definition above its first usage ---
def render_favorite(fav, idx):
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
    show_posters = st.session_state.get("show_posters", True)
    with cols[0]:
        if show_posters and fav.get("poster"):
            imdb_id_link = str(
                fav.get("imdb") or fav.get("imdb_id") or fav.get("imdbID") or ""
            ).strip()
            poster_url = fav["poster"]
            if imdb_id_link and imdb_id_link.startswith("tt"):
                st.markdown(
                    f'<a href="https://www.imdb.com/title/{imdb_id_link}/" target="_blank" rel="noopener">'
                    f'<img src="{poster_url}" width="120"/></a>',
                    unsafe_allow_html=True,
                )
            else:
                st.image(poster_url, width=120)
    with cols[1]:
        st.markdown(f"**{idx}. {fav['title']} ({fav['year']})** | ⭐ IMDb: {imdb_display} | 🍅 RT: {rt_display} | 🎯 CS: {fav.get('cineselectRating', 'N/A')}")
        # --- Comments section: İzlenecekler - match İzlenenler 6-step logic for add, edit, delete ---
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
                edit_mode_key = f"to_watch_comment_edit_mode_{fav['id']}_{c_idx}"
                if st.button("✏️", key=f"to_watch_comment_edit_{fav['id']}_{c_idx}"):
                    _safe_set_state(edit_mode_key, True)
                    st.rerun()
            with comment_row_cols[2]:
                if st.button("🗑️", key=f"comment_del_{fav['id']}_{c_idx}"):
                    new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                    # 1. Firestore update
                    db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                    # 2. Update fav["comments"]
                    fav["comments"] = new_comments
                    # 3. Update session_state
                    fav_type_val = fav.get("type", "").lower()
                    if fav_type_val in ["movie", "film"]:
                        target_list = st.session_state["favorite_movies"]
                    else:
                        target_list = st.session_state["favorite_series"]
                    for item in target_list:
                        if item.get("id") == fav["id"]:
                            item["comments"] = new_comments
                            break
                    # 4. Reset input state
                    comment_key = f"to_watch_comment_add_{fav['id']}"
                    comment_wb_key = f"to_watch_comment_add_wb_{fav['id']}"
                    _safe_set_state(comment_key, "")
                    _safe_set_state(comment_wb_key, "ss")
                    # 5. Success
                    st.success("🗑️ Yorum silindi!")
                    # 6. Rerun
                    st.rerun()
            # Inline edit UI if in edit mode
            if st.session_state.get(edit_mode_key, False):
                edit_text_key = f"to_watch_comment_edit_text_{fav['id']}_{c_idx}"
                edit_who_key = f"to_watch_comment_edit_who_{fav['id']}_{c_idx}"
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
                        ["öz", "ss", "öz❤️ss"],
                        index=(["öz", "ss", "öz❤️ss"].index(default_who) if default_who in ["öz", "ss", "öz❤️ss"] else 1),
                        key=edit_who_key
                    )
                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    # REPLACE WITH 6-STEP BLOCK
                    fav_type_val = fav.get("type", "").lower()
                    if fav_type_val in ["movie", "film"]:
                        target_list = st.session_state["favorite_movies"]
                    else:
                        target_list = st.session_state["favorite_series"]
                    if st.button("💾 Kaydet", key=f"comment_save_{fav['id']}_{c_idx}"):
                        now_str = format_turkish_datetime(_dt.now())
                        comments_sorted[c_idx] = {
                            "text": new_text.strip(),
                            "watchedBy": new_who,
                            "date": now_str,
                        }
                        # 1. Firestore update
                        db.collection("favorites").document(fav["id"]).update({"comments": comments_sorted})
                        # 2. Update fav["comments"]
                        fav["comments"] = comments_sorted
                        # 3. Update session_state
                        for item in target_list:
                            if item.get("id") == fav["id"]:
                                item["comments"] = comments_sorted
                                break
                        # 4. Reset edit mode
                        _safe_set_state(edit_mode_key, False)
                        # 5. Success
                        st.success("✏️ Yorum güncellendi!")
                        # 6. Rerun
                        st.rerun()
                with cancel_col:
                    if st.button("❌ İptal", key=f"to_watch_comment_cancel_{fav['id']}_{c_idx}"):
                        _safe_set_state(edit_mode_key, False)
                        st.rerun()
        # --- Yorum Ekle expander, İzlenenler-style, immediately after comments ---
        with st.expander("💬 Yorum Ekle"):
            comment_key = f"to_watch_comment_add_{fav['id']}"
            comment_wb_key = f"to_watch_comment_add_wb_{fav['id']}"
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
                ["öz", "ss", "öz❤️ss"],
                index=(["öz", "ss", "öz❤️ss"].index(st.session_state.get(comment_wb_key, "ss"))
                       if st.session_state.get(comment_wb_key, "ss") in ["öz", "ss", "öz❤️ss"] else 1),
                key=comment_wb_key,
                label_visibility="visible"
            )
            comment_btn_key = f"to_watch_comment_add_btn_{fav['id']}"
            comments_existing = fav.get("comments", [])
            new_comments = list(comments_existing) if comments_existing else []
            # Determine target_list before add
            fav_type_val = fav.get("type", "").lower()
            if fav_type_val in ["movie", "film"]:
                target_list = st.session_state["favorite_movies"]
            else:
                target_list = st.session_state["favorite_series"]
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
                    db.collection("favorites").document(fav["id"]).update({"comments": new_comments})
                    # 2. Update fav["comments"]
                    fav["comments"] = new_comments
                    # 3. Update session_state
                    for item in target_list:
                        if item.get("id") == fav["id"]:
                            item["comments"] = new_comments
                            break
                    # 4. Reset input states
                    _safe_set_state(comment_key, "")
                    _safe_set_state(comment_wb_key, "ss")
                    # 5. Success
                    st.success("💬 Yorum kaydedildi!")
                    # 6. Rerun
                    st.rerun()
    with cols[2]:
        with st.expander("✨ Options"):
            # --- (Comment edit/delete UI is now inline under the movie details, not in Options expander) ---
            # --- Status selectbox (short labels) and all action buttons grouped in expander ---
            status_options = ["to_watch", "öz", "ss", "öz❤️ss", "n/w", "🖤 BL"]
            # Compute current status string with new logic
            if fav.get("status") == "to_watch":
                current_status_str = "to_watch"
            elif fav.get("status") == "watched":
                wb = fav.get("watchedBy")
                if wb in ["öz", "ss", "öz❤️ss"]:
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
                key=f"status_{fav['id']}"
            )
            from datetime import datetime
            # --- Hızlı geçiş mantığı: İzlenecekler'de statü değişikliği anında diğer listeye aktar ---
            # Sadece "to_watch" listesindeyken hızlı geçiş uygula, onay ve yorum isteme
            fav_type = fav.get("type", "movie")
            if status_select != current_status_str:
                doc_ref = db.collection("favorites").document(fav["id"])
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
                        if item.get("id") == fav["id"]:
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
                elif status_select in ["öz", "ss", "öz❤️ss", "n/w"]:
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
                        if item.get("id") == fav["id"]:
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
                        if item.get("id") == fav["id"]:
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
            if st.button("✏️", key=f"edit_{fav['id']}"):
                _safe_set_state(f"edit_mode_{fav['id']}", True)
            # PIN FIRST: handle "Başa tuttur" BEFORE rendering input so it reflects new value immediately
            pin_now = st.button("📌 Başa tuttur", key=f"pin_{fav['id']}")
            if pin_now:
                # Find the maximum visible CS among current items and set to min(100, max_val + 1)
                try:
                    visible_cs = [
                        int(x.get("cineselectRating") or 0)
                        for x in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"])
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
                db.collection("favorites").document(fav["id"]).update({"cineselectRating": pin_val})
                # Update session_state as well
                for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                    if item.get("id") == fav["id"]:
                        item["cineselectRating"] = pin_val
                        break
                _safe_set_state(f"input_{fav['id']}", pin_val)
                st.success(f"📌 CineSelect puanı {pin_val} olarak güncellendi ve başa taşındı!")
                st.rerun()
            if st.session_state.get(f"edit_mode_{fav['id']}", False):
                i_key = f"input_{fav['id']}"
                current = _clamp_cs(fav.get("cineselectRating", 50))
                st.number_input(
                    "🎯 CS:",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.get(i_key, current),
                    step=1,
                    key=i_key
                )
                cols_edit = st.columns([1,2])
                with cols_edit[0]:
                    if st.button("✅ Kaydet", key=f"save_{fav['id']}"):
                        new_val = _clamp_cs(st.session_state.get(i_key, current))
                        db.collection("favorites").document(fav["id"]).update({"cineselectRating": new_val})
                        # Update session_state as well
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if item.get("id") == fav["id"]:
                                item["cineselectRating"] = new_val
                                break
                        st.success(f"✅ {fav['title']} güncellendi (CS={new_val}).")
                        _safe_set_state(f"edit_mode_{fav['id']}", False)
                        st.rerun()
                with cols_edit[1]:
                    st.caption("🔧 İpucu: 'Başa tuttur' butonuna bastıktan sonra 'Kaydet' ile kalıcılaştır.")

if fav_section == "📌 İzlenecekler":
    # Preserve pagination state; set default only if not present
    st.session_state.setdefault("to_watch_page", 1)
    page_size = 50
    # Determine which list to show based on media_type
    if media_type == "Movie":
        raw_list = st.session_state.get("favorite_movies", [])
    elif media_type == "TV Show":
        raw_list = st.session_state.get("favorite_series", [])
    else:
        raw_list = []

    # Only show to_watch items (missing/blank treated as to_watch)
    favorites = [f for f in raw_list if (f.get("status") in (None, "", "to_watch"))]

    # Sort according to current selection
    sort_name = st.session_state.get("fav_sort", "CineSelect")
    def _key_for(it: dict):
        try:
            if sort_name == "CineSelect":
                return int(it.get("cineselectRating") or 0)
            elif sort_name == "IMDb":
                v = it.get("imdbRating")
                return float(v) if v not in (None, "", "N/A") else 0.0
            elif sort_name == "RT":
                v = it.get("rt")
                return float(v) if v not in (None, "", "N/A") else 0.0
            elif sort_name == "Year":
                return int(str(it.get("year") or "0").strip() or 0)
        except Exception:
            return 0
        return 0
    favorites = sorted(favorites, key=_key_for, reverse=True)

    # Pagination: always include all previous pages, page starts from 1
    end_idx = st.session_state["to_watch_page"] * page_size
    favorites_to_render = favorites[:end_idx]
    for idx, fav in enumerate(favorites_to_render, start=1):
        render_favorite(fav, idx)
    if end_idx < len(favorites):
        if st.button("⬇️ Daha fazla yükle"):
            st.session_state["to_watch_page"] += 1
            st.rerun()
elif fav_section == "🎬 İzlenenler":
    st.markdown("---")
    sort_option_watched = st.selectbox(
        "Sort by (Watched):",
        ["Watched Date", "IMDb", "RT", "CineSelect", "Year"],
        index=0,
        key="watched_sort"
    )
    st.subheader(f"🎬 İzlenenler (sort: {sort_option_watched})")

    watched_docs = db.collection("favorites").where("status", "==", "watched").stream()
    watched_items = [doc.to_dict() for doc in watched_docs]

    # Varsayılan: Watched Date → yeniden eskiye
    if sort_option_watched == "Watched Date":
        watched_items = sorted(
            watched_items,
            key=lambda x: parse_turkish_or_iso_date(x.get("watchedAt")) or _dt.min,
            reverse=True
        )
    elif sort_option_watched == "IMDb":
        watched_items = sorted(watched_items, key=lambda x: float(x.get("imdbRating") or 0), reverse=True)
    elif sort_option_watched == "RT":
        watched_items = sorted(watched_items, key=lambda x: float(x.get("rt") or 0), reverse=True)
    elif sort_option_watched == "CineSelect":
        watched_items = sorted(watched_items, key=lambda x: int(x.get("cineselectRating") or 0), reverse=True)
    elif sort_option_watched == "Year":
        watched_items = sorted(watched_items, key=lambda x: int(x.get("year") or 0), reverse=True)

    # Sayfalama → İzlenecekler ile aynı mantık
    page_size = 50
    st.session_state.setdefault("watched_page", 1)
    end_idx = st.session_state["watched_page"] * page_size
    to_render = watched_items[:end_idx]

    for idx, fav in enumerate(to_render, start=1):
        render_favorite(fav, idx)

    if end_idx < len(watched_items):
        if st.button("⬇️ Daha fazla yükle (Watched)"):
            st.session_state["watched_page"] += 1
            st.rerun()
