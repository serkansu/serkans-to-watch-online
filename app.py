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
        {"file": "favorites.json", "owner": "serkansu", "repo": "serkans-to-watch-addon"},
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
    """Sort a flat media list by selected mode in descending order.
    mode: 'imdb' | 'cc' | 'year'
    """
    def key_fn(it):
        if mode == "imdb":
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
        # default: CineSelect score
        try:
            return int(it.get("cineselectRating") or 0)
        except Exception:
            return 0
    return sorted(items or [], key=key_fn, reverse=(mode != "cc"))

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

def sync_with_firebase(sort_mode="imdb"):
    favorites_data = {
        "movies": st.session_state.get("favorite_movies", []),
        "shows": st.session_state.get("favorite_series", [])
    }
    fix_invalid_imdb_ids(favorites_data)  # IMDb puanı olanları temizle
        # IMDb düzeltmesinden sonra type alanını normalize et
    for section in ["movies", "shows"]:
        for item in favorites_data[section]:
            t = item.get("type", "").lower()
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
    # ---- Apply export ordering
    sorted_movies = sort_flat_for_export(favorites_data.get("movies", []), sort_mode)
    sorted_series = sort_flat_for_export(favorites_data.get("shows", []), sort_mode)

    # Dışarı yazarken anahtar adını 'shows' -> 'series' olarak çevir
    output_data = {
        "movies": sorted_movies,
        "series": sorted_series,
    }
    with open("favorites.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        st.write("🔍 FAVORITES DEBUG (output):", output_data)
    st.success("✅ favorites.json dosyası yerel olarak oluşturuldu.")

    # GitHub'a push et
    push_favorites_to_github()

db = get_firestore()
# Firestore'dan verileri çek ve session'a yaz
movie_docs = db.collection("favorites").where("type", "==", "movie").stream()
series_docs = db.collection("favorites").where("type", "==", "show").stream()

st.session_state["favorite_movies"] = [doc.to_dict() for doc in movie_docs]
st.session_state["favorite_series"] = [doc.to_dict() for doc in series_docs]
st.set_page_config(page_title="Serkan’s To‑Watch Online", page_icon="🍿", layout="wide")

# --- Mobile Home Screen & Favicons ---
# High-res icons for iOS/Android home screen shortcuts and browser favicons.
ICON_180 = "https://em-content.zobj.net/source/apple/391/popcorn_1f37f.png"      # iOS (180x180)
ICON_192 = "https://em-content.zobj.net/source/microsoft-teams/363/popcorn_1f37f.png"  # Android (~192x192)
ICON_512 = "https://em-content.zobj.net/source/telegram/358/popcorn_1f37f.png"   # Android (~512x512)

st.markdown(
    f"""
    <link rel=\"apple-touch-icon\" sizes=\"180x180\" href=\"{ICON_180}\">\n
    <link rel=\"icon\" type=\"image/png\" sizes=\"192x192\" href=\"{ICON_192}\">\n
    <link rel=\"icon\" type=\"image/png\" sizes=\"512x512\" href=\"{ICON_512}\">\n
    <meta name=\"theme-color\" content=\"#111111\">\n
    """,
    unsafe_allow_html=True,
)
# --- /Mobile Home Screen & Favicons ---
st.markdown("<h1>🍿 Serkan'ın İzlenecek Film & Dizi Listesi <span style='color: orange'>ONLINE ✅</span></h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🏠 Go to Top"):
        st.rerun()

with col2:
    if "show_posters" not in st.session_state:
        st.session_state["show_posters"] = True

    if st.button("🖼️ Toggle Posters"):
        st.session_state["show_posters"] = not st.session_state["show_posters"]

    # Varsayılan sıralama modu: IMDb (yüksekten düşüğe)
    if "sync_sort_mode" not in st.session_state:
        st.session_state["sync_sort_mode"] = "imdb"

    if st.button("📂 JSON & CSV Sync"):
        sync_with_firebase(sort_mode=st.session_state.get("sync_sort_mode", "imdb"))
        st.success("✅ favorites.json ve seed_ratings.csv senkronize edildi.")

    # Butonun ALTINA üç radyo butonu (imdb, cc, year)
    st.radio(
        "Sync sıralaması",
        ["imdb", "cc", "year"],
        key="sync_sort_mode",
        horizontal=True,
        help="IMDb/Year: yüksekten düşüğe • CineSelect: küçükten büyüğe sıralar."
    )

def show_favorites_count():
    movie_docs = db.collection("favorites").where("type", "==", "movie").stream()
    series_docs = db.collection("favorites").where("type", "==", "show").stream()

    movie_count = len(list(movie_docs))
    series_count = len(list(series_docs))

    st.info(f"🎬 Favorite Movies: {movie_count} | 📺 Favorite TV Shows: {series_count}")
if st.button("📊 Favori Sayılarını Göster"):
    show_favorites_count()

show_posters = st.session_state["show_posters"]
media_type = st.radio("Search type:", ["Movie", "TV Show", "Actor/Actress"], horizontal=True)

# ---- Safe clear for search widgets (avoid modifying after instantiation)
if "clear_search" not in st.session_state:
    st.session_state.clear_search = False

if st.session_state.clear_search:
    # reset the flag and clear both the input widget's value and the session copy
    st.session_state.clear_search = False
    st.session_state["query_input"] = ""
    st.session_state.query = ""

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input(
    f"🔍 Search for a {media_type.lower()}",
    value=st.session_state.query,
    key="query_input",
)

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

            slider_key = f"stars_{item['id']}"
            manual_key = f"manual_{item['id']}"
            slider_val = st.slider("🎯 CineSelect Rating:", 1, 100, st.session_state.get(slider_key, 50), step=1, key=slider_key)
            manual_val = st.number_input("Manual value:", min_value=1, max_value=100, value=slider_val, step=1, key=manual_key)

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
                # 3) Firestore'a yaz
                db.collection("favorites").document(item["id"]).set({
                    "id": item["id"],
                    "title": item["title"],
                    "year": item.get("year"),
                    "imdb": imdb_id,
                    "poster": item.get("poster"),
                    "imdbRating": imdb_rating,                 # ✅ eklendi
                    "rt": rt_score,                            # ✅ CSV/OMDb’den gelen kesin değer
                    "cineselectRating": manual_val,
                    "type": media_key,
                })
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
st.subheader("❤️ İzlenecekler Listesi")
sort_option = st.selectbox("Sort by:", ["IMDb", "RT", "CineSelect", "Year"], index=0)
    
def get_sort_key(fav):
    try:
        if sort_option == "IMDb":
            return float(fav.get("imdbRating", 0) or 0)
        elif sort_option == "RT":
            return float(fav.get("rt", 0))
        elif sort_option == "CineSelect":
            return fav.get("cineselectRating", 0)
        elif sort_option == "Year":
            return int(fav.get("year", 0))
    except:
        return 0

def show_favorites(fav_type, label):
    docs = db.collection("favorites").where("type", "==", fav_type).stream()
    favorites = sorted(
        [doc.to_dict() for doc in docs],
        key=get_sort_key,
        reverse=(sort_option != "CineSelect")
    )

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
        cols = st.columns([1, 5, 1, 1])
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
        with cols[2]:
            if st.button("❌", key=f"remove_{fav['id']}"):
                db.collection("favorites").document(fav["id"]).delete()
                st.rerun()
        with cols[3]:
            if st.button("✏️", key=f"edit_{fav['id']}"):
                st.session_state[f"edit_mode_{fav['id']}"] = True

        if st.session_state.get(f"edit_mode_{fav['id']}", False):
            s_key = f"slider_{fav['id']}"
            i_key = f"input_{fav['id']}"
            current = _clamp_cs(fav.get("cineselectRating", 50))
            if s_key not in st.session_state:
                st.session_state[s_key] = current
            if i_key not in st.session_state:
                st.session_state[i_key] = current

            st.slider(
                "🎯 CS:", 1, 100, st.session_state[s_key], step=1,
                key=s_key, on_change=_sync_cs_from_slider, args=(s_key, i_key)
            )
            st.number_input(
                "CS (manuel):", min_value=1, max_value=100, value=st.session_state[i_key], step=1,
                key=i_key, on_change=_sync_cs_from_input, args=(i_key, s_key)
            )

            cols_edit = st.columns([1,1,2])
            with cols_edit[0]:
                if st.button("✅ Kaydet", key=f"save_{fav['id']}"):
                    new_val = _clamp_cs(st.session_state.get(i_key, st.session_state.get(s_key, current)))
                    db.collection("favorites").document(fav["id"]).update({"cineselectRating": new_val})
                    st.success(f"✅ {fav['title']} güncellendi.")
                    st.session_state[f"edit_mode_{fav['id']}"] = False
                    st.rerun()
            with cols_edit[1]:
                if st.button("📌 Başa tuttur", key=f"pin_{fav['id']}"):
                    # Aynı türdeki favorilerde en yüksek CS'yi bul, 10 ekle (üst sınır 100)
                    cur_max = 0
                    for d in db.collection("favorites").where("type", "==", fav_type).stream():
                        raw = (d.to_dict() or {}).get("cineselectRating")
                        try:
                            cs = int(raw)
                        except Exception:
                            cs = 0
                        if cs > cur_max:
                            cur_max = cs
                    pin_val = _clamp_cs(cur_max + 10)
                    try:
                        db.collection("favorites").document(fav["id"]).update({"cineselectRating": pin_val})
                    except Exception as e:
                        st.error(f"⚠️ Başa tutturma sırasında Firestore hatası: {e}")
                        return
                    _safe_set_state(s_key, pin_val)
                    _safe_set_state(i_key, pin_val)
                    st.success(f"📌 {fav['title']} en üste taşındı (CS={pin_val}).")
                    st.rerun()

if media_type == "Movie":
    show_favorites("movie", "Filmler")
elif media_type == "TV Show":
    show_favorites("show", "Diziler")

st.markdown("---")
if st.button("🔝 Go to Top Again"):
    st.rerun()

st.markdown("<p style='text-align: center; color: gray;'>Created by <b>SS</b></p>", unsafe_allow_html=True)
    
