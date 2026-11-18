import streamlit as st
import pandas as pd
import pickle
import requests
import urllib.parse
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Dark Green & Black Neon Styling
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 20%, #062614 0%, #03120a 50%, #010804 100%);
        color: #ffffff;
    }
    .stAppViewContainer {
        border: 3px solid #00ff66;
        box-shadow: inset 0 0 10px rgba(0, 255, 102, 0.25), 0 0 15px rgba(0, 255, 102, 0.35);
        border-radius: 12px;
        margin: 8px;
    }
    .center-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 14px rgba(0, 255, 102, 0.7);
        margin-bottom: 20px;
    }
    .top-right-logo {
        position: fixed;
        top: 20px;
        right: 25px;
        width: 65px;
        border-radius: 10px;
        z-index: 99999;
        box-shadow: 0 4px 15px rgba(0, 255, 102, 0.45);
    }
    .detail-box {
        background: #071910;
        border: 2px solid #00ff66;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 30px;
        box-shadow: 0 6px 25px rgba(0, 255, 102, 0.35);
    }
    .tag-badge {
        background: #0c2e1c;
        border: 1px solid #00ff66;
        color: #80ffb3;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 8px;
        line-height: 1.5;
        display: block;
    }
    /* Style buttons under posters to look like interactive neon triggers */
    div.stButton > button {
        width: 100%;
        background-color: #0b2919;
        color: #00ff66;
        border: 1px solid #00ff66;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background-color: #00ff66;
        color: #021208;
        box-shadow: 0 0 12px rgba(0, 255, 102, 0.8);
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)


def render_top_right_logo(image_filename="logo.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, image_filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{encoded_str}" class="top-right-logo">', unsafe_allow_html=True)


render_top_right_logo("logo.png")


@st.cache_resource
def load_data():
    movies_data = pickle.load(open('movies.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_data)

    # Automatically calculates similarity from movies.pkl (no similarity.pkl needed!)
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags'].fillna("")).toarray()
    sim = cosine_similarity(vectors)
    return movies_df, sim


movies, similarity = load_data()


@st.cache_data(show_spinner=False)
def fetch_full_movie_info(movie_id, movie_title, fallback_tags=""):
    headers = {"User-Agent": "Mozilla/5.0"}
    info = {
        "poster": None,
        "overview": "",
        "genres": "N/A",
        "director": "N/A",
        "cast": "N/A",
        "year": "N/A",
        "rating": "N/A"
    }

    # 1. OMDb API (Overview, Rating, Genre, Director, Cast)
    try:
        encoded_title = urllib.parse.quote(str(movie_title))
        omdb_url = f"https://www.omdbapi.com/?t={encoded_title}&apikey=trilogy"
        res = requests.get(omdb_url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if data.get("Response") == "True":
                if data.get("Poster") and data.get("Poster") != "N/A":
                    info["poster"] = data.get("Poster")
                if data.get("Plot") and data.get("Plot") != "N/A":
                    info["overview"] = data.get("Plot")
                if data.get("Genre") and data.get("Genre") != "N/A":
                    info["genres"] = data.get("Genre")
                if data.get("Director") and data.get("Director") != "N/A":
                    info["director"] = data.get("Director")
                if data.get("Actors") and data.get("Actors") != "N/A":
                    info["cast"] = data.get("Actors")
                if data.get("Year") and data.get("Year") != "N/A":
                    info["year"] = data.get("Year")
                if data.get("imdbRating") and data.get("imdbRating") != "N/A":
                    info["rating"] = data.get("imdbRating")
    except Exception:
        pass

    # 2. TMDB API Fallback
    if not info["poster"] or not info["overview"]:
        try:
            tmdb_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e1e67ae1295935638f7f555825ababc3&language=en-US"
            res = requests.get(tmdb_url, headers=headers, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if not info["poster"] and data.get("poster_path"):
                    info["poster"] = "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
                if not info["overview"] and data.get("overview"):
                    info["overview"] = data.get("overview")
                if info["genres"] == "N/A" and data.get("genres"):
                    info["genres"] = ", ".join([g["name"] for g in data["genres"]])
                if info["year"] == "N/A" and data.get("release_date"):
                    info["year"] = str(data.get("release_date"))[:4]
                if info["rating"] == "N/A" and data.get("vote_average"):
                    info["rating"] = str(round(data.get("vote_average"), 1))
        except Exception:
            pass

    # 3. Fallback to movies.pkl tags if overview is still empty
    if not info["overview"]:
        info["overview"] = str(fallback_tags).capitalize() if fallback_tags else "Overview not available."

    if not info["poster"]:
        safe_title = urllib.parse.quote(str(movie_title))
        info["poster"] = f"https://placehold.co/500x750/071910/00ff66/png?text={safe_title}"

    return info


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    scored_movies = []
    base_title = movie.split(':')[0].strip().lower()

    for idx, score in enumerate(distances):
        candidate_title = str(movies.iloc[idx].title).lower()
        if idx == movie_index:
            boosted_score = 100.0
        elif candidate_title.startswith(base_title):
            boosted_score = 50.0 + score
        else:
            boosted_score = score
        scored_movies.append((idx, boosted_score))

    # 8 Movies for 2 Rows x 4 Columns Grid
    movies_list = sorted(scored_movies, reverse=True, key=lambda x: x[1])[0:8]

    movie_args = []
    rec_items = []
    for i in movies_list:
        row = movies.iloc[i[0]]
        m_id = row.movie_id
        m_title = row.title
        m_tags = row['tags'] if 'tags' in movies.columns else "No tags available"
        rec_items.append({
            "id": m_id,
            "title": m_title,
            "tags": m_tags
        })
        movie_args.append((m_id, m_title, m_tags))

    with ThreadPoolExecutor(max_workers=8) as executor:
        rec_details = list(executor.map(lambda p: fetch_full_movie_info(*p), movie_args))

    return rec_items, rec_details


if "clicked_movie" not in st.session_state:
    st.session_state.clicked_movie = None

# --- UI Layout ---
st.markdown('<div class="center-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# Search box with label inside as placeholder and external label hidden
selected_movie_name = st.selectbox(
    "Search or Select a Movie:",
    movies['title'].values,
    index=None,
    placeholder="Search or Select a Movie...",
    label_visibility="collapsed"
)

# Reset clicked movie if user changes the dropdown search
if "last_searched" not in st.session_state or st.session_state.last_searched != selected_movie_name:
    st.session_state.last_searched = selected_movie_name
    st.session_state.clicked_movie = None

if selected_movie_name:
    with st.spinner(f"Loading posters for '{selected_movie_name}'..."):
        rec_items, rec_details = recommend(selected_movie_name)

        # Placeholder at the top where clicked movie details appear when a poster is clicked
        detail_container = st.container()

        # 4-Column Poster Grid (2 Rows x 4 Columns)
        for row_idx in range(2):
            cols = st.columns(4)
            for col_idx in range(4):
                idx = row_idx * 4 + col_idx
                item = rec_items[idx]
                info = rec_details[idx]

                with cols[col_idx]:
                    st.image(info["poster"], use_container_width=True)
                    if st.button(f"🎬 {item['title']}", key=f"poster_btn_{idx}"):
                        st.session_state.clicked_movie = {
                            "title": item["title"],
                            "tags": item["tags"],
                            "info": info
                        }

        # Render Overview, Rating & Tags ONLY when a movie poster button is clicked
        if st.session_state.clicked_movie:
            c_movie = st.session_state.clicked_movie
            c_info = c_movie["info"]

            with detail_container:
                st.markdown('<div class="detail-box">', unsafe_allow_html=True)
                col_p, col_d = st.columns([1, 3])

                with col_p:
                    st.image(c_info["poster"], use_container_width=True)
                    if st.button("✖ Close Details", key="close_btn"):
                        st.session_state.clicked_movie = None
                        st.rerun()

                with col_d:
                    st.markdown(f"## 🎬 {c_movie['title']} ({c_info['year']})")
                    st.markdown(
                        f"**⭐ IMDb Rating:** `{c_info['rating']} / 10` &nbsp;|&nbsp; **🎭 Genre:** `{c_info['genres']}`")
                    st.markdown(f"**🎥 Director:** {c_info['director']} &nbsp;|&nbsp; **🌟 Cast:** {c_info['cast']}")

                    st.markdown("#### 📖 Movie Overview")
                    st.write(c_info["overview"])

                    st.markdown("#### 🏷️ Movie Tags (`movies.pkl`)")
                    st.markdown(f'<div class="tag-badge">{str(c_movie["tags"])[:500]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)