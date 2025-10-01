import firebase_admin
from firebase_admin import credentials, firestore
import base64
import json
import os
import time
# Initialize Firebase only once
if not firebase_admin._apps:
    fb_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_B64")
    if fb_b64:
        try:
            fb_json = base64.b64decode(fb_b64).decode("utf-8")
            cred = credentials.Certificate(json.loads(fb_json))
        except Exception as e:
            raise RuntimeError(f"Invalid FIREBASE_SERVICE_ACCOUNT_JSON_B64: {e}")
    else:
        raise FileNotFoundError("Firebase key not found. Set FIREBASE_SERVICE_ACCOUNT_JSON_B64 in Render environment variables.")
    firebase_admin.initialize_app(cred)

db = firestore.client()
def safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

# Status options constant
STATUS_OPTIONS = ["to_watch", "öz", "ss", "öz❤️ss", "n/w", "🖤 BL"]
import time

# --- Comment rendering logic for all sections ---
def render_comments(fav, section="to_watch"):
    import streamlit as st
    from datetime import datetime as _dt
    # Helper for state
    def _safe_set_state(key, val):
        if key not in st.session_state or st.session_state[key] != val:
            st.session_state[key] = val
    db = globals().get("db")
    # Comments list
    comments = fav.get("comments", []) or []
    comments_sorted = sorted(comments, key=lambda c: parse_turkish_or_iso_date(c.get("date")), reverse=True)
    # Render each comment with edit/delete buttons
    for c_idx, c in enumerate(comments_sorted):
        text = c.get("text", "")
        who = c.get("watchedBy", "")
        date = c.get("date", "")
        fav_id = fav.get('id', '')
        comment_row_cols = st.columns([8, 1, 1])
        with comment_row_cols[0]:
            st.write(f"💬 {text} — ({who}) • {date}")
        with comment_row_cols[1]:
            edit_mode_key = f"{section}_comment_edit_mode_{fav_id}_{c_idx}"
            if st.button("✏️", key=f"{section}_comment_edit_{fav_id}_{c_idx}"):
                _safe_set_state(edit_mode_key, True)
                st.rerun()
        with comment_row_cols[2]:
            if st.button("🗑️", key=f"{section}_comment_del_{fav_id}_{c_idx}"):
                new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                db.collection("favorites").document(fav_id).update({"comments": new_comments})
                fav["comments"] = new_comments
                # Update session_state
                for item in (
                    st.session_state["favorite_movies"]
                    if (fav.get("type") or "movie") == "movie"
                    else st.session_state["favorite_series"]
                ):
                    if item.get("id", "") == fav_id:
                        item["comments"] = new_comments
                        break
                st.success("🗑️ Yorum silindi!")
                time.sleep(0.5)
                st.rerun()
        if st.session_state.get(edit_mode_key, False):
            edit_text_key = f"{section}_comment_edit_text_{fav_id}_{c_idx}"
            edit_who_key = f"{section}_comment_edit_who_{fav_id}_{c_idx}"
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
                if st.button("💾 Kaydet", key=f"{section}_comment_save_{fav_id}_{c_idx}"):
                    now_str = format_turkish_datetime(_dt.now())
                    comments_sorted[c_idx] = {
                        "text": new_text.strip(),
                        "watchedBy": new_who,
                        "date": now_str
                    }
                    db.collection("favorites").document(fav_id).update({"comments": comments_sorted})
                    fav["comments"] = comments_sorted
                    # update session_state
                    for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                        if item.get("id", "") == fav_id:
                            item["comments"] = comments_sorted
                            break
                    st.success("✏️ Yorum güncellendi!")
                    _safe_set_state(edit_mode_key, False)
                    time.sleep(0.5)
                    st.rerun()
            with cancel_col:
                if st.button("❌ İptal", key=f"{section}_comment_cancel_{fav_id}_{c_idx}"):
                    _safe_set_state(edit_mode_key, False)
                    st.rerun()
    # Add comment expander
    with st.expander("💬 Yorum Ekle"):
        fav_id = fav.get('id', '')
        comment_key = f"{section}_comment_add_{fav_id}"
        comment_wb_key = f"{section}_comment_add_wb_{fav_id}"
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
        comment_btn_key = f"{section}_comment_add_btn_{fav_id}"
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
                db.collection("favorites").document(fav_id).update({"comments": new_comments})
                fav["comments"] = new_comments
                for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie"
                             else st.session_state["favorite_series"]):
                    if item.get("id", "") == fav_id:
                        item["comments"] = new_comments
                        break
                _safe_set_state(comment_key, "")
                _safe_set_state(comment_wb_key, "ss")
                st.success("💬 Yorum kaydedildi!")
                time.sleep(0.5)
                st.rerun()
from tmdb import search_movie, search_tv, search_by_actor, search_person, search_by_person_id
from omdb import get_ratings
import csv
from pathlib import Path
# --- TMDB -> IMDb ID helper (cached) ---
def tmdb_imdb_id(item_id, media_type="movie"):
    """
    TMDB item ID'den (movie veya tv) IMDb ID döndürür.
    Sonuçları st.session_state içinde cache'ler.
    """
    import os, requests, streamlit as st
    cache_key = f"{media_type}:{item_id}"
    cache = st.session_state.setdefault("tmdb_imdb_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    api_key = os.getenv("TMDB_API_KEY")
    base = "https://api.themoviedb.org/3"
    path = "tv" if media_type == "tv" else "movie"
    url = f"{base}/{path}/{item_id}?api_key={api_key}&append_to_response=external_ids"

    try:
        res = requests.get(url, timeout=10).json()
        imdb_id = (res.get("external_ids") or {}).get("imdb_id")
    except Exception:
        imdb_id = None

    cache[cache_key] = imdb_id
    return imdb_id
# --- Helper functions for TMDb person credits ---
def get_actor_movies(person_id):
    import requests, os
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={api_key}&language=en-US"
    res = requests.get(url).json()
    return res.get("cast", [])

def get_director_movies(person_id):
    import requests, os
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={api_key}&language=en-US"
    res = requests.get(url).json()
    crew = res.get("crew", [])
    return [c for c in crew if c.get("job") == "Director"]

# --- Arama Bölümü ---
import streamlit as st

# Varsayalım ki bu fonksiyonlar zaten tanımlı:
# search_movie(query), search_person(query, job=None), search_by_person_id(person_id)

# --- Arama kutusu ---
st.markdown("---")
st.header("🔍 Arama")
query = st.text_input("Film, kişi veya yönetmen ara", key="main_search_query")
if query:
    search_type = st.radio(
        "Ne aramak istersiniz?",
        ["Movies", "People", "Directors"],
        horizontal=True,
        key="search_type"
    )

    if search_type == "Movies":
        movies = search_movie(query)
        # En yeni yıldan eskiye sırala
        movies = sorted(
            movies,
            key=lambda m: (m.get("release_date") or m.get("first_air_date") or ""),
            reverse=True
        )
        if movies:
            movie_titles = [f"{m.get('title') or m.get('name')} ({(m.get('release_date') or m.get('first_air_date') or '')[:4]})" for m in movies]
            idx = st.selectbox("Film seçin", range(len(movies)), format_func=lambda i: movie_titles[i], key="search_movie_select")
            selected_movie = movies[idx]
            st.write("Seçilen film:", selected_movie.get("title") or selected_movie.get("name"))
            # Burada film detaylarını gösterebilirsiniz...
        else:
            st.info("Sonuç bulunamadı.")

    elif search_type == "People":
        persons = search_person(query)
        if persons:
            person_names = [f"{p.get('name')} ({p.get('known_for_department','')})" for p in persons]
            idx = st.selectbox("Kişi seçin", range(len(persons)), format_func=lambda i: person_names[i], key="search_person_select")
            selected_person = persons[idx]
            st.write("Seçilen kişi:", selected_person.get("name"))

            # Rol seçimi: Actor veya Director
            role = st.radio("Rol seçiniz", ["Actor", "Director"], horizontal=True, key="role_select")

            if role == "Actor":
                credits = get_actor_movies(selected_person.get("id"))
            else:
                credits = get_director_movies(selected_person.get("id"))

            credits = sorted(credits, key=lambda m: (m.get("release_date") or m.get("first_air_date") or ""), reverse=True)
            st.markdown("**Projeler:**")

            selected_items = []
            for c in credits:
                title = c.get("title") or c.get("name")
                year = (c.get("release_date") or c.get("first_air_date") or "????")[:4]
                poster_path = c.get("poster_path")
                media_type = "tv" if c.get("media_type") == "tv" else "movie"
                # imdb_id = tmdb_imdb_id(c["id"], media_type=media_type)  # Removed lazy fetch

                cols = st.columns([1, 5])
                with cols[0]:
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                        st.image(poster_url, width=100)
                with cols[1]:
                    if st.checkbox(f"{title} ({year})", key=f"chk_{c['id']}"):
                        selected_items.append({
                            "id": str(c["id"]),
                            "title": title,
                            "year": year,
                            "poster": f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None,
                            "imdb": None,
                            "type": "movie" if media_type == "movie" else "show"
                        })

            # Bulk add button is always visible; validate selection on click
            if st.button("➕ Hepsini Favorilere Ekle", key="add_people_bulk"):
                if not selected_items:
                    st.warning("⚠️ Önce birkaç proje seçmelisiniz.")
                else:
                    for item in selected_items:
                        _media_type_for_tmdb = "tv" if item.get("type") == "show" else "movie"
                        imdb_id = tmdb_imdb_id(int(item["id"]), media_type=_media_type_for_tmdb)
                        db.collection("favorites").document(item["id"]).set({
                            "id": item["id"],
                            "title": item["title"],
                            "year": item["year"],
                            "poster": item["poster"],
                            "imdb": imdb_id or "",
                            "type": "tv" if item.get("type") == "show" else "movie",
                            "status": "to_watch",
                            "cineselectRating": 100,
                            "imdbRating": 0.0,
                            "rt": 0,
                        })
                    st.success(f"✅ {len(selected_items)} proje favorilere eklendi (CS=100).")
                    st.rerun()
        else:
            st.info("Sonuç bulunamadı.")

    elif search_type == "Directors":
        # search_person(query, job="Director") fonksiyonu varsa kullan, yoksa filtrele
        try:
            directors = search_person(query, job="Director")
        except TypeError:
            # Eğer job parametresi yoksa, kendimiz filtreleyelim
            persons_all = search_person(query)
            directors = [p for p in persons_all if p.get("known_for_department") == "Directing"]
        if directors:
            director_names = [f"{d.get('name')} ({d.get('known_for_department','')})" for d in directors]
            idx = st.selectbox("Yönetmen seçin", range(len(directors)), format_func=lambda i: director_names[i], key="search_director_select")
            selected_director = directors[idx]
            st.write("Seçilen yönetmen:", selected_director.get("name"))
            credits = get_director_movies(selected_director.get("id"))
            credits = sorted(credits, key=lambda m: (m.get("release_date") or m.get("first_air_date") or ""), reverse=True)
            st.markdown("**Yönetmenlik yaptığı filmler/diziler:**")

            selected_items = []
            for c in credits:
                title = c.get("title") or c.get("name")
                year = (c.get("release_date") or c.get("first_air_date") or "????")[:4]
                poster_path = c.get("poster_path")
                media_type = "tv" if c.get("media_type") == "tv" else "movie"
                # imdb_id = tmdb_imdb_id(c["id"], media_type=media_type)  # Removed lazy fetch

                cols = st.columns([1, 5])
                with cols[0]:
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                        st.image(poster_url, width=100)
                with cols[1]:
                    if st.checkbox(f"{title} ({year})", key=f"chk_{c['id']}"):
                        selected_items.append({
                            "id": str(c["id"]),
                            "title": title,
                            "year": year,
                            "poster": f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None,
                            "imdb": None,
                            "type": "movie" if media_type == "movie" else "show"
                        })

            # Bulk add button is always visible; validate selection on click
            if st.button("➕ Hepsini Favorilere Ekle", key="add_directors_bulk"):
                if not selected_items:
                    st.warning("⚠️ Önce birkaç proje seçmelisiniz.")
                else:
                    for item in selected_items:
                        _media_type_for_tmdb = "tv" if item.get("type") == "show" else "movie"
                        imdb_id = tmdb_imdb_id(int(item["id"]), media_type=_media_type_for_tmdb)
                        db.collection("favorites").document(item["id"]).set({
                            "id": item["id"],
                            "title": item["title"],
                            "year": item["year"],
                            "poster": item["poster"],
                            "imdb": imdb_id or "",
                            "type": "tv" if item.get("type") == "show" else "movie",
                            "status": "to_watch",
                            "cineselectRating": 100,
                            "imdbRating": 0.0,
                            "rt": 0,
                        })
                    st.success(f"✅ {len(selected_items)} proje favorilere eklendi (CS=100).")
                    st.rerun()
        else:
            st.info("Sonuç bulunamadı.")
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
                        ir_val = float(ir) if ir not in (None, "", "N/A") else 0.0
                    except Exception:
                        ir_val = 0.0
                    try:
                        rt_val = int(float(rt)) if rt not in (None, "", "N/A") else 0
                    except Exception:
                        rt_val = 0
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

# --- Cached Firestore favorites loader ---
@st.cache_data(ttl=240)  # Cache Firestore favorites for 240 seconds
def load_favorites():
    global db
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
            item["type"] = "movie"
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
# --- /Quick Toolbar ---


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
media_type = st.radio("Search type:", ["Movie", "TV Show", "Actor/Actress", "Director", "Writer/Creator"], horizontal=True)

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
# --- Kişi & film arama önerileri ---
if query:
    # Önce kişi araması yap
    persons = search_person(query)
    if persons:
        person_names = [
            f"{p.get('name')} (id: {p.get('id')})" for p in persons
        ]
        selected_person = st.selectbox(
            "🎭 Bulunan kişiler:", ["Seçiniz..."] + person_names,
            key="person_select"
        )
        if selected_person != "Seçiniz...":
            # Seçilen kişiye ait filmleri/TV’leri getir
            person_id = int(selected_person.split("id:")[1].strip(") "))
            credits = search_by_person_id(person_id)
            if credits:
                credit_titles = [
                    f"{c.get('title') or c.get('name')} ({(c.get('release_date') or c.get('first_air_date') or '')[:4]})"
                    for c in credits
                ]
                st.write("🎬 Bu kişinin projeleri:", credit_titles)

    # Her durumda film araması da yap
    movies = search_movie(query)
    if movies:
        movie_options = [
            f"{m.get('title')} ({(m.get('release_date') or m.get('first_air_date') or '')[:4]})"
            for m in movies
        ]
        # Filmleri dropdown olarak gösterelim
        selected_movie = st.selectbox(
            "🎬 Bulunan filmler:",
            ["Seçiniz..."] + movie_options,
            key="movie_select"
        )
        if selected_movie != "Seçiniz...":
            st.success(f"Seçtiğiniz film: {selected_movie}")
    else:
        st.info("Hiç film bulunamadı.")
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
            cs = int(fav.get("cineselectRating", "N/A") or 0)
            try:
                imdb = float(fav.get("imdbRating", 0.0) or 0)
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
            return float(fav.get("rt", 0) or 0)
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
def _add_item_to_favorites(item, media_key, cs_value=100):
    """Add one item to Firestore favorites with IMDb/RT enrichment and given CineSelect score."""
    from omdb import get_ratings, fetch_ratings  # ensure fetch_ratings is available
        # 🔥 Daha önce eklenmiş mi kontrolü
    item_id = item.get("id", "")
    item_title = item.get("title", "")
    item_year = item.get("year", "")
    item_poster = item.get("poster", None)
    # check if already exists
    _doc_ref = db.collection("favorites").document(item_id)
    _prev = _doc_ref.get()
    if _prev.exists:
        st.warning(f"⚠️ {item_title} zaten favorilerde var!", icon="⚠️")
        return
    # 1) Ensure IMDb ID
    imdb_id = (item.get("imdb", "") or "").strip()
    if not imdb_id or imdb_id == "tt0000000":
        imdb_id = get_imdb_id_from_tmdb(
            title=item_title,
            year=item_year,
            is_series=(media_key == "show"),
        )

    # 2) IMDb/RT ratings (prefer local CSV; then OMDb by ID; then OMDb by Title/Year)
    stats = {}
    raw_id = {}
    raw_title = {}
    # a) CSV
    seed_hit = read_seed_rating(imdb_id)
    if seed_hit and (seed_hit.get("imdb_rating") or seed_hit.get("rt")):
        stats = {"imdb_rating": seed_hit.get("imdb_rating"), "rt": seed_hit.get("rt")}
    # b) OMDb by ID
    if not stats and imdb_id:
        stats = get_ratings(imdb_id) or {}
        raw_id = (stats.get("raw") or {})
    # c) OMDb by Title/Year
    if not stats or ((stats.get("imdb_rating") in (None, 0, "N/A")) and (stats.get("rt") in (None, 0, "N/A"))):
        ir, rt, raw_title = fetch_ratings(item_title, item_year)
        stats = {"imdb_rating": ir, "rt": rt}

    imdb_rating = float(stats.get("imdb_rating") or 0.0)
    rt_score    = int(stats.get("rt") or 0)

    # 3) Firestore write (preserve previous addedAt if exists)
    _doc_ref = db.collection("favorites").document(item_id)
    _prev = _doc_ref.get()
    _prev_data = _prev.to_dict() if _prev.exists else {}
    _added_at = _prev_data.get("addedAt") or firestore.SERVER_TIMESTAMP

    payload = {
        "id": item_id,
        "title": item_title,
        "year": item_year,
        "imdb": imdb_id,
        "poster": item_poster,
        "imdbRating": imdb_rating,
        "rt": rt_score,
        "cineselectRating": int(cs_value) if cs_value is not None else 100,
        "type": media_key,
        "addedAt": _added_at,
    }
    _doc_ref.set(payload)

    # 4) Ensure seed_ratings.csv has this row
    append_seed_rating(
        imdb_id=imdb_id,
        title=item_title,
        year=item_year,
        imdb_rating=imdb_rating,
        rt_score=rt_score,
    )

if query:
    st.session_state.query = query
    if media_type == "Movie":
        results = search_movie(query)
    elif media_type == "TV Show":
        results = search_tv(query)
    elif media_type == "Actor/Actress":
        results = search_by_actor(query)
    elif media_type == "Director":
        from tmdb import search_by_director
        results = search_by_director(query)
    elif media_type == "Writer/Creator":
        from tmdb import search_by_writer
        results = search_by_writer(query)

    try:
        results = sorted(results, key=lambda x: x.get("cineselectRating", 0), reverse=True)
    except:
        pass

    if not results:
        st.error("❌ No results found.")
    else:
        selected_items = []
        for idx, item in enumerate(results):
            st.divider()
            item_id = item.get("id", "")
            item_title = item.get("title", "")
            item_year = item.get("year", "—")
            item_poster = item.get("poster", None)
            if item_poster and show_posters:
                imdb_id_link = str(
                    item.get("imdb", "")
                    or item.get("imdb_id", "")
                    or item.get("imdbID", "")
                    or ""
                ).strip()
                poster_url = item_poster
                if imdb_id_link.startswith("tt"):
                    st.markdown(
                        f'<a href="https://www.imdb.com/title/{imdb_id_link}/" target="_blank" rel="noopener">'
                        f'<img src="{poster_url}" width="180"/></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(poster_url, width=180)

            st.markdown(f"**{idx+1}. {item_title} ({item_year})**")

            # Bulk-selection checkbox
            _sel_key = f"sel_{item_id}_{idx}"
            if st.checkbox("Seç", key=_sel_key):
                _media_key_bulk = "movie" if media_type == "Movie" else ("show" if media_type == "TV Show" else "movie")
                selected_items.append((item, _media_key_bulk))

            # --- warn inline if this exact title+year already exists in favorites ---
            _item_key = f"{_norm_title(item_title)}::{str(item_year or '')}"
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

            # IMDb rating display: prefer explicit imdbRating; if not present, use numeric `imdb`
            _imdb_rating_field = item.get("imdbRating", None)
            if isinstance(_imdb_rating_field, (int, float)):
                imdb_display = f"{float(_imdb_rating_field):.1f}" if _imdb_rating_field > 0 else "N/A"
            elif isinstance(item.get("imdb", None), (int, float)):
                imdb_display = f"{float(item.get('imdb', 0)):.1f}" if item.get("imdb", 0) > 0 else "N/A"
            else:
                imdb_display = "N/A"

            rt_val = item.get("rt", 0)
            rt_display = f"{int(rt_val)}%" if isinstance(rt_val, (int, float)) and rt_val > 0 else "N/A"
            st.markdown(f"⭐ IMDb: {imdb_display} &nbsp;&nbsp; 🍅 RT: {rt_display}", unsafe_allow_html=True)

            manual_key = f"manual_{item_id}"
            manual_val = st.number_input(
                "🎯 CineSelect Rating:",
                min_value=1,
                max_value=100,
                value=st.session_state.get(manual_key, 50),
                step=1,
                key=manual_key
            )

            if st.button("Add to Favorites", key=f"btn_{item_id}"):
                media_key = "movie" if media_type == "Movie" else ("show" if media_type == "TV Show" else "movie")
                # Use the manual per-item CS value for single add
                _add_item_to_favorites(item, media_key, cs_value=manual_val)
                st.success(f"✅ {item_title} added to favorites!")
                st.session_state.clear_search = True
                st.toast("Refreshing…", icon="🔄")
                time.sleep(1.2)
                st.rerun()

        # --- Bulk add for selected items (default CS=100) ---
        if selected_items and st.button("➕ Add Selected to Favorites", key="btn_add_selected_bulk"):
            for _it, _mkey in selected_items:
                _add_item_to_favorites(_it, _mkey, cs_value=100)
            st.success(f"✅ {len(selected_items)} öğe favorilere eklendi (CS=100).")
            st.toast("Refreshing…", icon="🔄")
            # 🔥 Yeni eklenen öğelerin ekranda görünmesi için cache temizle ve tekrar yükle
            st.cache_data.clear()
            movies, shows = load_favorites()
            st.session_state["favorite_movies"] = movies
            st.session_state["favorite_series"] = shows

            # 🔹 Arama kutusunu temizle
            st.session_state.query = ""
            st.session_state.query_input = ""
            st.session_state.clear_search = True

            # 🔹 Ana ekrana dön
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
    # Fetch from Firestore only if favorites not provided; filter by to_watch/None/""
    if favorites is None:
        to_watch_docs = db.collection("favorites").where("type", "==", fav_type).stream()
        favorites = [
            doc.to_dict() for doc in to_watch_docs
            if doc.to_dict().get("status") in ("to_watch", None, "")
        ]
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
        fav_id = fav.get('id', '')
        fav_title = fav.get('title', '')
        fav_year = fav.get('year', '')
        imdb_display = (
            f"{float(fav.get('imdbRating', 0.0) or 0):.1f}"
            if fav.get('imdbRating', 0.0) not in (None, "", "N/A") and isinstance(fav.get('imdbRating', 0.0), (int, float))
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
                    fav.get("imdb", "") or fav.get("imdb_id", "") or fav.get("imdbID", "") or ""
                ).strip()
                poster_url = fav.get("poster")
                if imdb_id_link and imdb_id_link.startswith("tt"):
                    st.markdown(
                        f'<a href="https://www.imdb.com/title/{imdb_id_link}/" target="_blank" rel="noopener">'
                        f'<img src="{poster_url}" width="120"/></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(poster_url, width=120)
        with cols[1]:
            st.markdown(f"**{idx+1}. {fav_title} ({fav_year})** | ⭐ IMDb: {imdb_display} | 🍅 RT: {rt_display} | 🎯 CS: {fav.get('cineselectRating', 'N/A')}")
            # --- Comments section: İzlenenler-style logic, with sort, edit, delete, and add ---
            comments = fav.get("comments", [])
            from datetime import datetime as _dt
            comments_sorted = sorted(comments, key=lambda c: parse_turkish_or_iso_date(c.get("date")), reverse=True)
            for c_idx, c in enumerate(comments_sorted):
                fav_id = fav.get('id', '')
                text = c.get("text", "")
                who = c.get("watchedBy", "")
                date = c.get("date", "")
                comment_row_cols = st.columns([8, 1, 1])
                with comment_row_cols[0]:
                    st.write(f"💬 {text} — ({who}) • {date}")
                with comment_row_cols[1]:
                    edit_mode_key = f"to_watch_comment_edit_mode_{fav_id}_{c_idx}"
                    if st.button("✏️", key=f"to_watch_comment_edit_{fav_id}_{c_idx}"):
                        _safe_set_state(edit_mode_key, True)
                        st.rerun()
                with comment_row_cols[2]:
                    if st.button("🗑️", key=f"to_watch_comment_del_{fav_id}_{c_idx}"):
                        new_comments = [x for j, x in enumerate(comments_sorted) if j != c_idx]
                        db.collection("favorites").document(fav_id).update({"comments": new_comments})
                        fav["comments"] = new_comments
                        # update session_state immediately after Firestore update (mirror İzlenenler)
                        for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                            if item.get("id", "") == fav_id:
                                item["comments"] = new_comments
                                break
                        st.success("🗑️ Yorum silindi!")
                        st.rerun()
                # Inline edit UI if in edit mode
                if st.session_state.get(edit_mode_key, False):
                    edit_text_key = f"to_watch_comment_edit_text_{fav_id}_{c_idx}"
                    edit_who_key = f"to_watch_comment_edit_who_{fav_id}_{c_idx}"
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
                        if st.button("💾 Kaydet", key=f"to_watch_comment_save_{fav_id}_{c_idx}"):
                            now_str = format_turkish_datetime(_dt.now())
                            comments_sorted[c_idx] = {
                                "text": new_text.strip(),
                                "watchedBy": new_who,
                                "date": now_str
                            }
                            db.collection("favorites").document(fav_id).update({"comments": comments_sorted})
                            fav["comments"] = comments_sorted
                            # update session_state immediately after Firestore update (mirror İzlenenler)
                            for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie" else st.session_state["favorite_series"]):
                                if item.get("id", "") == fav_id:
                                    item["comments"] = comments_sorted
                                    break
                            st.success("✏️ Yorum güncellendi!")
                            _safe_set_state(edit_mode_key, False)
                            st.rerun()
                    with cancel_col:
                        if st.button("❌ İptal", key=f"to_watch_comment_cancel_{fav_id}_{c_idx}"):
                            _safe_set_state(edit_mode_key, False)
                            st.rerun()
            # --- Yorum Ekle expander, İzlenenler-style, immediately after comments ---
            with st.expander("💬 Yorum Ekle"):
                fav_id = fav.get('id', '')
                comment_key = f"to_watch_comment_add_{fav_id}"
                comment_wb_key = f"to_watch_comment_add_wb_{fav_id}"
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
                comment_btn_key = f"to_watch_comment_add_btn_{fav_id}"
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
                        db.collection("favorites").document(fav_id).update({"comments": new_comments})
                        # 2. Update fav["comments"]
                        fav["comments"] = new_comments
                        # 3. session_state güncellemesi
                        for item in (st.session_state["favorite_movies"] if (fav.get("type") or "movie") == "movie"
                                     else st.session_state["favorite_series"]):
                            if item.get("id", "") == fav_id:
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
                status_options = STATUS_OPTIONS
                # Compute current status string with new logic
                if fav.get("status", "") == "to_watch":
                    current_status_str = "to_watch"
                elif fav.get("status", "") == "watched":
                    wb = fav.get("watchedBy", "")
                    if wb in ["öz", "ss", "öz❤️ss"]:
                        current_status_str = wb
                    else:
                        current_status_str = "n/w"
                elif fav.get("status", "") == "blacklist":
                    current_status_str = "🖤 BL"
                else:
                    current_status_str = "to_watch"
                status_select = st.selectbox(
                    "Watched by",
                    status_options,
                    index=status_options.index(current_status_str) if current_status_str in status_options else 0,
                    key=f"status_{fav_id}"
                )
                from datetime import datetime
                # --- Hızlı geçiş mantığı: İzlenecekler'de statü değişikliği anında diğer listeye aktar ---
                # Sadece "to_watch" listesindeyken hızlı geçiş uygula, onay ve yorum isteme
                if status_select != current_status_str:
                    fav_id = fav.get('id', '')
                    doc_ref = db.collection("favorites").document(fav_id)
                    now_str = format_turkish_datetime(datetime.now())
                    # Status transitions: always set all relevant fields for consistency
                    if status_select == "to_watch":
                        doc_ref.update({
                            "status": "to_watch",
                            "watchedBy": None,
                            "watchedAt": None,
                            "watchedEmoji": None,
                            "blacklistedBy": None,
                            "blacklistedAt": None,
                            "cineselectRating": None,
                        })
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if item.get("id", "") == fav_id:
                                item.update({
                                    "status": "to_watch",
                                    "watchedBy": None,
                                    "watchedAt": None,
                                    "watchedEmoji": None,
                                    "blacklistedBy": None,
                                    "blacklistedAt": None,
                                    "cineselectRating": None,
                                })
                                break
                        st.session_state["fav_section"] = "📌 İzlenecekler"
                        st.success(f"✅ {fav.get('title', '')} durumu güncellendi: to_watch")
                        time.sleep(0.5)
                        st.rerun()
                    elif status_select in ["öz", "ss", "öz❤️ss", "n/w"]:
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
                            if item.get("id", "") == fav_id:
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
                        st.success(f"✅ {fav.get('title', '')} durumu güncellendi: watched ({status_select})")
                        time.sleep(0.5)
                        st.rerun()
                    elif status_select == "🖤 BL":
                        doc_ref.update({
                            "status": "blacklist",
                            "blacklistedBy": "🖤 BL",
                            "blacklistedAt": now_str,
                            "watchedBy": None,
                            "watchedAt": None,
                            "watchedEmoji": None,
                            "cineselectRating": None,
                        })
                        for item in (st.session_state["favorite_movies"] if fav_type == "movie" else st.session_state["favorite_series"]):
                            if item.get("id", "") == fav_id:
                                item.update({
                                    "status": "blacklist",
                                    "blacklistedBy": "🖤 BL",
                                    "blacklistedAt": now_str,
                                    "watchedBy": None,
                                    "watchedAt": None,
                                    "watchedEmoji": None,
                                    "cineselectRating": None,
                                })
                                break
                        st.session_state["fav_section"] = "🖤 Blacklist"
                        st.success(f"✅ {fav.get('title', '')} durumu güncellendi: blacklist")
                        time.sleep(0.5)
                        st.rerun()
