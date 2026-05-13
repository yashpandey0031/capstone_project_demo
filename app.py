from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MOVIELENS_DIR = DATA_DIR / "ml-1m"
MOVIELENS_ZIP = DATA_DIR / "ml-1m.zip"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
PPO_MODEL_PATH = APP_DIR / "ppo_recommender.zip"
DQN_MODEL_PATH = APP_DIR / "dqn_recommender.zip"


st.set_page_config(
    page_title="RL Movie Recommender Simulator",
    page_icon="🎬",
    layout="wide",
)


def ensure_movielens_data() -> bool:
    ratings_path = MOVIELENS_DIR / "ratings.dat"
    movies_path = MOVIELENS_DIR / "movies.dat"
    if ratings_path.exists() and movies_path.exists():
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not MOVIELENS_ZIP.exists():
            urllib.request.urlretrieve(MOVIELENS_URL, MOVIELENS_ZIP)
        with zipfile.ZipFile(MOVIELENS_ZIP, "r") as archive:
            archive.extractall(DATA_DIR)
        return ratings_path.exists() and movies_path.exists()
    except (urllib.error.URLError, OSError, zipfile.BadZipFile):
        return False


def build_fallback_catalog() -> pd.DataFrame:
    rows = [
        (1, "Inception", "Action|Sci-Fi|Thriller", 4.4, 98),
        (2, "The Dark Knight", "Action|Crime|Drama", 4.6, 96),
        (3, "Interstellar", "Adventure|Drama|Sci-Fi", 4.5, 92),
        (4, "Parasite", "Drama|Thriller", 4.4, 88),
        (5, "The Matrix", "Action|Sci-Fi", 4.5, 95),
        (6, "Pulp Fiction", "Crime|Drama", 4.3, 90),
        (7, "Memento", "Mystery|Thriller", 4.2, 82),
        (8, "The Prestige", "Drama|Mystery|Thriller", 4.2, 84),
        (9, "Her", "Drama|Romance|Sci-Fi", 4.1, 76),
        (10, "Spirited Away", "Animation|Adventure|Family", 4.5, 91),
        (11, "Toy Story", "Animation|Adventure|Comedy", 4.2, 93),
        (12, "The Lion King", "Animation|Adventure|Drama", 4.3, 94),
        (13, "Gladiator", "Action|Adventure|Drama", 4.1, 87),
        (14, "Whiplash", "Drama|Music", 4.4, 79),
        (15, "The Godfather", "Crime|Drama", 4.7, 99),
    ]
    catalog = pd.DataFrame(rows, columns=["movie_id", "title", "genres", "avg_rating", "ratings_count"])
    catalog["popularity_score"] = catalog["ratings_count"] / catalog["ratings_count"].max()
    catalog["source"] = "fallback"
    return catalog


@st.cache_data(show_spinner=False)
def load_catalog() -> tuple[pd.DataFrame, str]:
    if ensure_movielens_data():
        ratings = pd.read_csv(
            MOVIELENS_DIR / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        movies = pd.read_csv(
            MOVIELENS_DIR / "movies.dat",
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
        movie_stats = (
            ratings.groupby("movie_id")
            .agg(avg_rating=("rating", "mean"), ratings_count=("rating", "size"))
            .reset_index()
        )
        catalog = movies.merge(movie_stats, on="movie_id", how="left")
        catalog["avg_rating"] = catalog["avg_rating"].fillna(0.0)
        catalog["ratings_count"] = catalog["ratings_count"].fillna(0).astype(int)
        max_count = max(float(catalog["ratings_count"].max()), 1.0)
        catalog["popularity_score"] = catalog["ratings_count"] / max_count
        catalog["source"] = "MovieLens 1M"
        catalog = catalog.sort_values(["ratings_count", "avg_rating"], ascending=False).reset_index(drop=True)
        return catalog, "MovieLens 1M"

    return build_fallback_catalog(), "built-in fallback demo"


def normalize_genre_string(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def build_user_profile(catalog: pd.DataFrame, selected_movie_ids: list[int]) -> dict:
    selected = catalog[catalog["movie_id"].isin(selected_movie_ids)]
    genre_counts = Counter()
    for genres in selected["genres"].tolist():
        genre_counts.update(normalize_genre_string(genres))

    favorite_genres = [genre for genre, _ in genre_counts.most_common(3)]
    selected_popularity = float(selected["popularity_score"].mean()) if not selected.empty else 0.4
    selected_rating = float(selected["avg_rating"].mean()) if not selected.empty else 3.5

    return {
        "selected_movie_ids": set(selected_movie_ids),
        "favorite_genres": favorite_genres,
        "selected_popularity": selected_popularity,
        "selected_rating": selected_rating,
    }


def compute_reward(clicked: bool, watch_pct: float, stayed: bool, alpha: float = 1.0, beta: float = 0.5, gamma: float = 2.0) -> float:
    return alpha * int(clicked) + beta * watch_pct + gamma * int(stayed)


def estimate_affinity(movie_row: pd.Series, profile: dict) -> float:
    genres = normalize_genre_string(movie_row["genres"])
    if not genres:
        return 0.0

    overlap = len(set(genres).intersection(profile["favorite_genres"])) / max(len(genres), 1)
    popularity = float(movie_row["popularity_score"])
    rating = float(movie_row["avg_rating"])
    rating_norm = min(max(rating / 5.0, 0.0), 1.0)
    diversity_bonus = 1.0 - min(len(set(genres).intersection(profile["favorite_genres"])), 3) / 3.0
    affinity = 0.45 * overlap + 0.25 * popularity + 0.20 * rating_norm + 0.10 * diversity_bonus
    return float(np.clip(affinity, 0.0, 1.0))


def simulate_user_response(profile: dict, movie_row: pd.Series, rng: np.random.Generator) -> tuple[bool, float, bool, float]:
    affinity = estimate_affinity(movie_row, profile)
    if affinity >= 0.82:
        return True, float(rng.uniform(0.85, 1.0)), True, affinity
    if affinity >= 0.68:
        return True, float(rng.uniform(0.65, 0.92)), True, affinity
    if affinity >= 0.50:
        return True, float(rng.uniform(0.35, 0.70)), bool(rng.random() > 0.22), affinity
    if affinity >= 0.35:
        return True, float(rng.uniform(0.12, 0.40)), bool(rng.random() > 0.55), affinity
    return False, 0.0, False, affinity


def score_candidates(catalog: pd.DataFrame, profile: dict, watched_ids: set[int], algorithm: str) -> pd.DataFrame:
    candidates = catalog[~catalog["movie_id"].isin(watched_ids)].copy()
    if candidates.empty:
        return candidates

    candidates["affinity"] = candidates.apply(lambda row: estimate_affinity(row, profile), axis=1)

    if algorithm == "CTR-Based":
        candidates["policy_score"] = 0.90 * candidates["popularity_score"] + 0.10 * candidates["affinity"]
    elif algorithm == "DQN Agent":
        candidates["policy_score"] = (
            0.55 * candidates["affinity"]
            + 0.25 * candidates["avg_rating"].clip(lower=0, upper=5) / 5.0
            + 0.20 * candidates["popularity_score"]
        )
    else:
        candidates["policy_score"] = (
            0.45 * candidates["affinity"]
            + 0.20 * candidates["popularity_score"]
            + 0.35 * (1.0 - candidates["genres"].apply(lambda value: len(set(normalize_genre_string(value)).intersection(profile["favorite_genres"]))) / 3.0)
        )

    return candidates.sort_values(["policy_score", "avg_rating", "ratings_count"], ascending=False).reset_index(drop=True)


def pick_recommendation(catalog: pd.DataFrame, profile: dict, watched_ids: set[int], algorithm: str, rng: np.random.Generator) -> pd.Series:
    ranked = score_candidates(catalog, profile, watched_ids, algorithm)
    if ranked.empty:
        return catalog.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

    if algorithm == "PPO Agent":
        top_pool = ranked.head(min(10, len(ranked))).copy()
        weights = np.array(top_pool["policy_score"].clip(lower=0.01).tolist(), dtype=float)
        weights = weights / weights.sum()
        choice = int(rng.choice(top_pool.index.to_numpy(), p=weights))
        return top_pool.loc[choice]

    return ranked.iloc[0]


def run_session(
    catalog: pd.DataFrame,
    selected_movie_ids: list[int],
    algorithm: str,
    steps: int,
    seed: int | None = None,
    trained_models: dict[str, object] | None = None,
) -> tuple[list[dict], dict]:
    profile = build_user_profile(catalog, selected_movie_ids)
    watched_ids = set(selected_movie_ids)
    rng = np.random.default_rng(seed)
    item_list = build_item_list(catalog)
    history = encode_history(selected_movie_ids, item_list)
    records: list[dict] = []

    for step_number in range(1, steps + 1):
        recommendation, action, used_model = choose_recommendation(
            catalog,
            profile,
            watched_ids,
            algorithm,
            rng,
            trained_models=trained_models,
            item_list=item_list,
            history=history,
        )
        clicked, watch_pct, stayed, affinity = simulate_user_response(profile, recommendation, rng)
        reward = compute_reward(clicked, watch_pct, stayed)

        records.append(
            {
                "step": step_number,
                "movie_id": int(recommendation["movie_id"]),
                "title": recommendation["title"],
                "genres": recommendation["genres"],
                "affinity": round(float(affinity), 3),
                "source": "trained checkpoint" if used_model else "heuristic policy",
                "clicked": clicked,
                "watch_pct": round(float(watch_pct), 3),
                "stayed": stayed,
                "reward": round(float(reward), 3),
            }
        )

        watched_ids.add(int(recommendation["movie_id"]))
        history = history[1:] + [action]
        if not stayed:
            break

    summary = {
        "total_reward": round(float(sum(item["reward"] for item in records)), 3),
        "session_length": len(records),
        "click_rate": round(float(np.mean([1 if item["clicked"] else 0 for item in records])) if records else 0.0, 3),
        "avg_watch_pct": round(float(np.mean([item["watch_pct"] for item in records])) if records else 0.0, 3),
        "stayed_through_session": bool(records and records[-1]["stayed"]),
    }
    return records, summary


def compare_algorithms(catalog: pd.DataFrame, selected_movie_ids: list[int], steps: int, seed: int | None = None, trained_models: dict[str, object] | None = None) -> pd.DataFrame:
    rows = []
    for algorithm in ["CTR-Based", "DQN Agent", "PPO Agent"]:
        records, summary = run_session(catalog, selected_movie_ids, algorithm, steps, seed=seed, trained_models=trained_models)
        rows.append(
            {
                "algorithm": algorithm,
                "total_reward": summary["total_reward"],
                "session_length": summary["session_length"],
                "click_rate": summary["click_rate"],
                "avg_watch_pct": summary["avg_watch_pct"],
                "ended_early": not summary["stayed_through_session"],
                "top_recommendation": records[0]["title"] if records else "-",
                "policy_mode": records[0].get("source", "-") if records else "-",
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_trained_models() -> tuple[dict[str, object], list[str]]:
    try:
        from stable_baselines3 import DQN, PPO
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {}, [f"Stable-Baselines3 is not available: {exc}"]

    loaded_models: dict[str, object] = {}
    status_messages: list[str] = []

    for algorithm, model_path, model_cls in [
        ("DQN Agent", DQN_MODEL_PATH, DQN),
        ("PPO Agent", PPO_MODEL_PATH, PPO),
    ]:
        if not model_path.exists():
            status_messages.append(f"{model_path.name} was not found.")
            continue

        try:
            loaded_models[algorithm] = model_cls.load(model_path, device="cpu")
            status_messages.append(f"Loaded {algorithm} from {model_path.name}.")
        except Exception as exc:  # pragma: no cover - checkpoint compatibility depends on env
            status_messages.append(f"Failed to load {model_path.name}: {exc}")

    return loaded_models, status_messages


def build_item_list(catalog: pd.DataFrame, n_items: int = 200) -> list[int]:
    return (
        catalog.sort_values(["ratings_count", "avg_rating"], ascending=False)["movie_id"]
        .drop_duplicates()
        .head(n_items)
        .tolist()
    )


def encode_history(movie_ids: list[int], item_list: list[int], history_len: int = 5) -> list[int]:
    item_index = {movie_id: index for index, movie_id in enumerate(item_list)}
    encoded = [item_index.get(movie_id, 0) for movie_id in movie_ids]
    if len(encoded) < history_len:
        encoded = [0] * (history_len - len(encoded)) + encoded
    return encoded[-history_len:]


def choose_recommendation(
    catalog: pd.DataFrame,
    profile: dict,
    watched_ids: set[int],
    algorithm: str,
    rng: np.random.Generator,
    trained_models: dict[str, object] | None = None,
    item_list: list[int] | None = None,
    history: list[int] | None = None,
) -> tuple[pd.Series, int, bool]:
    trained_models = trained_models or {}
    model = trained_models.get(algorithm)
    item_list = item_list or build_item_list(catalog)
    history = history or encode_history(sorted(watched_ids), item_list)

    if model is not None:
        obs = np.array(history, dtype=np.float32)
        deterministic = algorithm == "DQN Agent"
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        if action < 0 or action >= len(item_list):
            action = int(rng.integers(0, len(item_list)))

        movie_id = item_list[action]
        recommendation = catalog[catalog["movie_id"] == movie_id]
        if not recommendation.empty:
            return recommendation.iloc[0], action, True

    ranked = score_candidates(catalog, profile, watched_ids, algorithm)
    if ranked.empty:
        fallback = catalog.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        fallback_action = item_list.index(int(fallback["movie_id"])) if int(fallback["movie_id"]) in item_list else 0
        return fallback, fallback_action, False

    if algorithm == "PPO Agent":
        top_pool = ranked.head(min(10, len(ranked))).copy()
        weights = np.array(top_pool["policy_score"].clip(lower=0.01).tolist(), dtype=float)
        weights = weights / weights.sum()
        choice = int(rng.choice(top_pool.index.to_numpy(), p=weights))
        recommendation = top_pool.loc[choice]
    else:
        recommendation = ranked.iloc[0]

    action = item_list.index(int(recommendation["movie_id"])) if int(recommendation["movie_id"]) in item_list else 0
    return recommendation, action, False


def algorithm_explanation(algorithm: str) -> str:
    if algorithm == "CTR-Based":
        return "Popularity-first baseline: recommends the MovieLens items with the strongest overall engagement signal, then lightly adjusts for the selected watch history."
    if algorithm == "DQN Agent":
        return "DQN policy: uses the provided trained checkpoint when available, otherwise falls back to a MovieLens-based long-term reward heuristic."
    return "PPO policy: uses the provided trained checkpoint when available, otherwise falls back to a MovieLens-based stochastic heuristic."


def format_movie_option(row: pd.Series) -> str:
    return f"{row['title']}  ·  {row['genres']}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(28,34,48,0.96), rgba(16,18,27,0.96));
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 60px rgba(0,0,0,0.18);
        }
        .hero h1 {
            margin-bottom: 0.2rem;
            font-size: 2.2rem;
        }
        .hero p {
            margin-top: 0;
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    catalog, source_label = load_catalog()
    trained_models, model_status = load_trained_models()

    st.markdown(
        """
        <div class="hero">
            <h1>RL Movie Recommendation Simulator</h1>
            <p>Choose a watch history from MovieLens 1M, compare CTR-based, DQN, and PPO policies, and watch the simulated user response step by step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Data source: {source_label}. The app keeps the reward logic from your notebook but runs as an interactive demo.")
    if trained_models:
        st.success("Loaded the provided trained checkpoints for DQN and PPO.")
    else:
        st.warning("Could not load the trained checkpoints, so DQN and PPO will fall back to heuristic policies.")

    with st.expander("Model loading status"):
        for message in model_status:
            st.write(message)

    top_catalog = catalog.head(30).copy()
    top_catalog["label"] = top_catalog.apply(format_movie_option, axis=1)

    with st.form("simulation_form"):
        st.subheader("Step 1 - Your watch history")
        selected_labels = st.multiselect(
            "Pick 2-4 movies you've watched",
            options=top_catalog["label"].tolist(),
            default=top_catalog["label"].tolist()[:3],
        )

        st.subheader("Step 2 - Choose algorithm")
        algorithm = st.radio(
            "How should the system decide what to recommend?",
            options=["CTR-Based", "DQN Agent", "PPO Agent"],
            horizontal=True,
            index=1,
        )

        steps = st.slider("Number of recommendation steps", min_value=3, max_value=10, value=5, step=1)
        run_button = st.form_submit_button("Start recommendation session")

    st.write(algorithm_explanation(algorithm))

    label_to_id = dict(zip(top_catalog["label"], top_catalog["movie_id"]))
    selected_movie_ids = [int(label_to_id[label]) for label in selected_labels if label in label_to_id]

    if len(selected_movie_ids) < 2:
        st.warning("Pick at least 2 movies to start the simulation.")
        return

    selected_movies = top_catalog[top_catalog["movie_id"].isin(selected_movie_ids)][["title", "genres", "avg_rating"]]
    st.write("Your watch history preview")
    st.dataframe(selected_movies, use_container_width=True, hide_index=True)

    if run_button:
        records, summary = run_session(catalog, selected_movie_ids, algorithm, steps, seed=42, trained_models=trained_models)
        comparison = compare_algorithms(catalog, selected_movie_ids, steps, seed=42, trained_models=trained_models)

        st.subheader("Step 3 - Run the session")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total reward", f"{summary['total_reward']:.3f}")
        col2.metric("Session length", summary["session_length"])
        col3.metric("Click rate", f"{summary['click_rate']:.0%}")
        col4.metric("Avg watch", f"{summary['avg_watch_pct']:.0%}")

        session_df = pd.DataFrame(records)
        st.write("Session trace")
        st.dataframe(
            session_df[["step", "title", "genres", "affinity", "source", "clicked", "watch_pct", "stayed", "reward"]],
            use_container_width=True,
            hide_index=True,
        )

        st.write("Algorithm comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("algorithm")[["total_reward", "click_rate", "avg_watch_pct"]])

        st.write("What the selected policy produced")
        for row in records:
            with st.expander(f"Step {row['step']}: {row['title']}"):
                st.write(f"Genres: {row['genres']}")
                st.write(f"Affinity score: {row['affinity']}")
                st.write(f"Clicked: {row['clicked']}")
                st.write(f"Watch percentage: {row['watch_pct']:.0%}")
                st.write(f"Stayed in session: {row['stayed']}")
                st.write(f"Reward: {row['reward']:.3f}")

    with st.expander("How this maps to the notebook"):
        st.write(
            "The notebook's reward equation is preserved: click + watch time + session continuation. "
            "The Streamlit app uses the real MovieLens 1M catalog and rating aggregates, then replaces the notebook's heavy RL training cells with lightweight policy simulators so the demo runs interactively."
        )
        st.code(
            "reward = alpha * int(clicked) + beta * watch_pct + gamma * int(stayed)",
            language="python",
        )

    with st.expander("What is actually trained here?"):
        st.write(
            "CTR stays as a baseline policy. DQN and PPO now load your provided checkpoints when Stable-Baselines3 is available. "
            "If a checkpoint cannot be loaded in the current environment, the app falls back to a MovieLens-based heuristic policy so the demo still works."
        )


if __name__ == "__main__":
    main()