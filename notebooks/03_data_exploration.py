# %% [markdown]
# # 03. Исследование датасета курсовой: признаки VK и психологические черты
#
# Только анализ данных, модели курсовой здесь не строятся. Выводы и таблицы
# собраны в `docs/data_exploration.md`.
#
# Это реальные персональные данные, поэтому:
# - файл читается построчно, `vk_raw` выбрасывается сразу после извлечения;
# - в кеш (`data/exploration_table.csv.gz`, не в git) попадают только числа и
#   длины строк: без vk_id (вместо него номер человека), полных дат рождения,
#   текстов и ссылок. Числовые коды города и вуза остаются: они нужны исходному набору;
# - на экран и в `results/` выводятся только агрегаты. Максимумы не печатаются,
#   вместо них 99-й перцентиль; категории с малым числом людей скрываются.
#
# Запуск целиком: `uv run python notebooks/03_data_exploration.py`.
# Первый запуск извлекает таблицу (несколько минут), дальше берётся кеш.
# Тяжёлые таблицы (нелинейные пары, повторная CV, перестановочные тесты) тоже берутся
# из `results/*.csv`, если они есть. Пересчёт: `EXPLORE_RECOMPUTE=1`.

# %%
import gc
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.set_option("display.width", 200, "display.max_rows", 300, "display.max_columns", 30)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA_PATH = ROOT / "data" / "dataset_1000.jsonl"
CACHE_PATH = ROOT / "data" / "exploration_table.csv.gz"
RESULTS = ROOT / "results"

TARGETS = {
    "target_extraversion_introversion": "extra",
    "target_attachment_detachment": "attach",
    "target_self_control_impulsivity": "control",
    "target_emotional_stability_instability": "emo",
    "target_expressiveness_practicality": "express",
}
# Имена шкал внутри test_raw_result_json
RAW_SCALE_PREFIX = {
    "Экстраверсия": "extra",
    "Привязанность": "attach",
    "Самоконтроль": "control",
    "Эмоциональная": "emo",
    "Экспрессивность": "express",
}
FREE_TEXT = ["about", "activities", "books", "games", "interests", "movies", "music",
             "quotes", "tv", "status", "home_town"]
JSON_LISTS = ["career", "schools", "universities", "military", "docs", "gifts", "group_names",
              "photo_urls", "wall_comments", "post_likers", "video_titles"]
SECTIONS = ["wall", "groups", "subscriptions", "photos", "videos"]
DAY = 86400
# Московское время: часовой пояс людей неизвестен, это допущение
MSK = 3 * 3600


def safe_len(s):
    return len(s) if isinstance(s, str) else np.nan


def json_len(s):
    try:
        return len(json.loads(s))
    except (TypeError, ValueError):
        return np.nan


def birth_year(bdate):
    # В формате VK год может быть скрыт: «Д.М» вместо «Д.М.ГГГГ»
    m = re.fullmatch(r"\d{1,2}\.\d{1,2}\.(\d{4})", bdate) if isinstance(bdate, str) else None
    return int(m.group(1)) if m else np.nan


def wall_features(items, owner, t_test):
    """Интерпретируемые признаки стены. Из текста берётся только длина."""
    out = {"w_n": len(items)}
    if not items:
        return out
    dates = np.array(sorted(p.get("date", 0) for p in items))
    is_repost = np.array([bool(p.get("copy_history")) for p in items])
    text_len = np.array([len(p.get("text") or "") for p in items])
    likes = np.array([(p.get("likes") or {}).get("count", 0) for p in items], float)
    comments = np.array([(p.get("comments") or {}).get("count", 0) for p in items], float)
    raw_dates = np.array([p.get("date", 0) for p in items])
    hours = ((raw_dates + MSK) % DAY) // 3600
    before = raw_dates < t_test
    span_days = max((dates[-1] - dates[0]) / DAY, 1.0)
    gaps = np.diff(dates) / DAY
    original = ~is_repost
    out.update(
        w_first=dates[0],
        w_last=dates[-1],
        w_span_days=span_days,
        w_posts_per_year=len(items) / span_days * 365.25 if len(items) > 1 else np.nan,
        w_repost_share=is_repost.mean(),
        w_text_share=(text_len > 0).mean(),
        w_median_text_len=np.median(text_len),
        w_orig_median_text_len=np.median(text_len[original]) if original.any() else np.nan,
        w_others_share=np.mean([p.get("from_id") != owner for p in items]),
        w_night_share=np.mean((hours >= 0) & (hours < 6)),
        w_gap_median_days=np.median(gaps) if len(gaps) else np.nan,
        w_gap_cv=gaps.std() / gaps.mean() if len(gaps) > 1 and gaps.mean() > 0 else np.nan,
        w_median_likes=np.median(likes),
        w_median_comments=np.median(comments),
        w_n_before_test=int(before.sum()),
        w_n_12m_before=int(((raw_dates >= t_test - 365 * DAY) & before).sum()),
        w_n_12m_after=int(((raw_dates >= t_test) & (raw_dates < t_test + 365 * DAY)).sum()),
        w_median_likes_before=np.median(likes[before]) if before.any() else np.nan,
        w_repost_share_before=is_repost[before].mean() if before.any() else np.nan,
    )
    return out


def group_features(items, prefix):
    out = {}
    if not items:
        return out
    members = [g["members_count"] for g in items if isinstance(g.get("members_count"), int)]
    types = [g.get("type") for g in items]
    out[f"{prefix}_n_kept"] = len(items)
    out[f"{prefix}_median_log_members"] = np.median(np.log1p(members)) if members else np.nan
    out[f"{prefix}_admin_share"] = np.mean([bool(g.get("is_admin")) for g in items])
    out[f"{prefix}_profile_share"] = np.mean([t == "profile" for t in types])
    out[f"{prefix}_page_share"] = np.mean([t == "page" for t in types])
    return out


def extract_row(rec, person_of, string_fields):
    vk_id = rec["vk_id"]
    person = person_of.setdefault(vk_id, len(person_of))
    row = {
        "person": person,
        "vk_id_int": vk_id,  # заменяется рангом перед записью в кеш
        "completion_date": rec["completion_date"],
        "test_short_name_len": safe_len(rec.get("test_short_name")),
    }
    for k in TARGETS:
        row[k] = rec.get(k)
    for k in ("test_answer_text_total_len", "test_events_accepted_count",
              "test_events_count", "test_items_count"):
        row[k] = rec.get(k)

    # Служебная копия результата теста: сверяем баллы и структуру
    try:
        raw = json.loads(rec["test_raw_result_json"])
        row["raw_n_entries"] = len(raw)
        row["raw_n_answer_entries"] = sum("q_id" in e for e in raw)
        for e in raw:
            for k, v in e.items():
                for pref, short in RAW_SCALE_PREFIX.items():
                    if k.startswith(pref):
                        row[f"raw_{short}"] = v
    except (TypeError, ValueError, AttributeError):
        row["raw_n_entries"] = np.nan

    # Готовые признаки: числа как есть, у строк только наличие и длина
    vf = rec["vk_features"]
    for k, v in vf.items():
        if isinstance(v, str):
            string_fields.add(k)
        elif isinstance(v, bool):
            row[k] = int(v)
        elif isinstance(v, (int, float)) or v is None:
            row[k] = v
    for name in FREE_TEXT:
        row[f"len_{name}"] = safe_len(vf.get(f"vk_{name}"))
    row["len_text_blob_check"] = safe_len(vf.get("vk_text_blob"))
    for name in JSON_LISTS:
        row[f"n_{name}"] = json_len(vf.get(f"vk_{name}_json"))
    for name in ("occupation_name", "bdate", "city_title", "activity"):
        row[f"has_{name}"] = int(vf.get(f"vk_{name}") is not None)
    row["bdate_year"] = birth_year(vf.get("vk_bdate"))
    # Категории без персональных значений (закрытый список VK)
    row["cat_profile_visibility"] = vf.get("vk_profile_visibility")
    row["cat_occupation_type"] = vf.get("vk_occupation_type")
    for name in ("friends", "groups", "videos", "wall", "photos", "profile", "docs",
                 "gifts", "followers", "subscriptions", "wall_comments", "likers"):
        row[f"err_{name}"] = int(vf.get(f"vk_{name}_error") is not None)

    # Сырой профиль: только то, чего нет в готовых признаках
    vr = rec["vk_raw"]
    prof = vr.get("profile") or {}
    row["raw_profile_id_match"] = int(prof.get("id") == vk_id)
    counters = prof.get("counters") or {}
    for c in ("mutual_friends", "clips", "clips_followers", "clips_likes", "clips_views",
              "video_playlists", "wishes"):
        row[f"rc_{c}"] = counters.get(c)
    personal = prof.get("personal") or {}
    row["p_n_personal_fields"] = len(personal)
    row["p_n_langs"] = len(personal.get("langs") or [])
    for c in ("alcohol", "smoking", "life_main", "people_main"):
        row[f"p_{c}"] = personal.get(c)
    for section in SECTIONS:
        blob = vr.get(section) or {}
        err = blob.get("error")
        row[f"errcode_{section}"] = err.get("code") if isinstance(err, dict) else np.nan
        row[f"rawcount_{section}"] = blob.get("count")

    # completion_date хранится в миллисекундах, даты постов VK в секундах
    t_test = rec["completion_date"] / 1000
    row.update(wall_features((vr.get("wall") or {}).get("items") or [], vk_id, t_test))
    row.update(group_features((vr.get("groups") or {}).get("items") or [], "g"))
    row.update(group_features((vr.get("subscriptions") or {}).get("items") or [], "s"))

    photos = (vr.get("photos") or {}).get("items") or []
    if photos:
        pd_dates = [p["date"] for p in photos if isinstance(p.get("date"), int)]
        row["ph_first"] = min(pd_dates) if pd_dates else np.nan
        row["ph_last"] = max(pd_dates) if pd_dates else np.nan
        row["ph_median_likes"] = np.median([(p.get("likes") or {}).get("count", 0) for p in photos])
        row["ph_tags_share"] = np.mean([bool(p.get("has_tags")) for p in photos])
        row["ph_n_kept"] = len(photos)
    videos = (vr.get("videos") or {}).get("items") or []
    if videos:
        vd_dates = [v["date"] for v in videos if isinstance(v.get("date"), int)]
        row["v_first"] = min(vd_dates) if vd_dates else np.nan
        row["v_last"] = max(vd_dates) if vd_dates else np.nan
    return row


def build_table():
    """Читает файл построчно и держит в памяти только одну запись."""
    rows, person_of, string_fields = [], {}, set()
    with open(DATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            del line
            rows.append(extract_row(rec, person_of, string_fields))
            del rec
            if i % 100 == 0:
                gc.collect()
                print(f"  прочитано записей: {i}", flush=True)
    table = pd.DataFrame(rows)
    # vk_id растёт со временем регистрации: сохраняем только его перцентиль
    table["vk_id_rank"] = table["vk_id_int"].rank(pct=True)
    # Строковые поля: в таблице остались только их None, значения не сохраняем
    return table.drop(columns=["vk_id_int", *(string_fields & set(table.columns))])


if CACHE_PATH.exists():
    df = pd.read_csv(CACHE_PATH)
else:
    df = build_table()
    df.to_csv(CACHE_PATH, index=False)
print("Таблица:", df.shape)

# %% [markdown]
# ## 1. Базовые величины и проверка целей
#
# Дата сбора в данных не записана, берём её по последнему `last_seen_time`.
# Возраст: год рождения виден не у всех, значения вне 14–70 лет считаем пропуском.

# %%
TARGET_COLS = list(TARGETS)
SHORT = list(TARGETS.values())
df = df.rename(columns=TARGETS)
RESULTS.mkdir(exist_ok=True)
COLLECT_T = df["vk_last_seen_time"].max()
test_dt = pd.to_datetime(df["completion_date"], unit="ms")
df["test_year"] = test_dt.dt.year
df["gap_years"] = (COLLECT_T - df["completion_date"] / 1000) / DAY / 365.25
age = df["test_year"] - df["bdate_year"]
print("Год рождения виден:", df["bdate_year"].notna().mean(),
      "| возраст вне 14–70:", int(((age < 14) | (age > 70)).sum()))
df["age"] = age.where(age.between(14, 70))
df["pilot"] = (df["raw_n_answer_entries"] == 1).astype(int)

print("Записей и людей:", len(df), df["person"].nunique())
print("Год теста:", df["test_year"].value_counts().sort_index().to_dict())
print("Разрыв тест–сбор, лет:", df["gap_years"].describe().round(2).to_dict())
print("Пол (1/2):", df["vk_sex"].value_counts(normalize=True).round(3).to_dict())
print("Баллы совпадают с test_raw_result_json:",
      {s: float((df[f"raw_{s}"] == df[s]).mean()) for s in SHORT})

scale_stats = df[SHORT].agg(["mean", "std", "median", "skew", "kurt"]).T.round(2)
scale_stats["min"] = df[SHORT].min()
scale_stats["max"] = df[SHORT].max()
scale_stats["share_at_bounds"] = ((df[SHORT] == 15) | (df[SHORT] == 75)).mean().round(3)
print(scale_stats)
print("Почти плоский профиль (sd по 5 чертам < 2):", int((df[SHORT].std(axis=1) < 2).sum()))
scale_stats.to_csv(RESULTS / "scale_stats.csv")

# %% [markdown]
# ## 2. Качество признаков

# %%
raw_all = [c for c in df if c.startswith("vk_") and c != "vk_id_rank"]
quality = pd.DataFrame({
    "na": df[raw_all].isna().mean(),
    "unique": df[raw_all].nunique(),
    "zero": (df[raw_all] == 0).mean(),
    "p50": df[raw_all].median(),
    "p99": df[raw_all].quantile(0.99),
    "skew": df[raw_all].skew(),
}).round(3)
quality.to_csv(RESULTS / "feature_quality.csv")
empty_or_const = quality.index[quality["unique"] <= 1].tolist()
print(f"Числовых полей vk_features: {len(raw_all)}, пустых или константных: {len(empty_or_const)}")
print(empty_or_const)
RAW_FEATURES = [c for c in raw_all if c not in empty_or_const]
print("Исходный набор:", len(RAW_FEATURES))

print("\nЛимиты сбора, доля записей ровно на лимите:")
for col, cap in [("vk_wall_posts_collected", 200), ("vk_photos_collected", 300),
                 ("vk_videos_collected", 100), ("vk_photos_kept", 20),
                 ("vk_groups_collected", 1000), ("vk_followers_collected", 5000)]:
    print(f"  {col} = {cap}: {(df[col] == cap).mean():.3f}; выше лимита: {(df[col] > cap).mean():.3f}")

print("\nДубли (доля совпадений / ρ Спирмена):")
for a, b in [("vk_counter_friends", "vk_friends_total"), ("vk_counter_followers", "vk_followers_total"),
             ("vk_counter_videos", "vk_videos_total"), ("vk_counter_photos", "vk_photos_total"),
             ("vk_groups_total", "vk_groups_collected"), ("vk_subscriptions_total", "vk_subscriptions_collected"),
             ("vk_counter_pages", "vk_subscriptions_total"), ("vk_counter_groups", "vk_groups_total"),
             ("vk_wall_posts_total", "vk_wall_posts_collected"), ("vk_post_avg_comments", "vk_post_max_comments"),
             ("vk_groups_total", "vk_text_blob_len")]:
    pair = df[[a, b]].dropna()
    print(f"  {a} ~ {b}: n={len(pair)}, равны {(pair[a] == pair[b]).mean():.3f}, "
          f"ρ={stats.spearmanr(pair[a], pair[b])[0]:.3f}")
print("  followers_count_profile = followers + friends:",
      round(float((df["vk_followers_count_profile"] == df["vk_counter_followers"] + df["vk_counter_friends"]).mean()), 3))

print("\nЗаглушки:")
for col in ["vk_graduation", "vk_university", "vk_faculty", "vk_relation"]:
    print(f"  {col}: пропуск {df[col].isna().mean():.3f}, ноль {(df[col] == 0).mean():.3f}, значение {(df[col] > 0).mean():.3f}")
print("  vk_friends_collected = 0 при ошибке сбора:", int(((df["vk_friends_collected"] == 0) & (df["err_friends"] == 1)).sum()),
      "из", int((df["err_friends"] == 1).sum()))
print("  posts_with_photo > собранных постов:", round(float((df["vk_posts_with_photo_attachments"] > df["vk_wall_posts_collected"]).mean()), 3))

print("\nОшибки сбора по позиции записи в файле (доля по децилям строк):")
decile = pd.qcut(df.index, 10, labels=False)
for col in ["err_friends", "err_groups", "err_videos"]:
    print(f"  {col}: {df[col].groupby(decile).mean().round(2).tolist()}")
print("Коды ошибок API: groups", df["errcode_groups"].value_counts().to_dict(),
      "videos", df["errcode_videos"].value_counts().to_dict())
print("ρ(номер строки, last_seen_time):", round(stats.spearmanr(df.index, df["vk_last_seen_time"], nan_policy="omit")[0], 3))
print("ρ(перцентиль vk_id, первый пост):", round(stats.spearmanr(df["vk_id_rank"], df["w_first"])[0], 3))
print("Все профили открыты:", df["cat_profile_visibility"].value_counts().to_dict(),
      "| без постов:", int((df["vk_wall_posts_collected"] == 0).sum()))
print("Зависимость максимумов от числа собранных постов (ρ):",
      {c: round(stats.spearmanr(df[c], df["vk_wall_posts_collected"])[0], 2)
       for c in ["vk_post_max_likes", "vk_post_max_views", "vk_post_max_text_len", "vk_post_avg_likes", "vk_post_avg_views"]})

# %% [markdown]
# ## 3. Наборы признаков
#
# Решения приняты только по качеству данных, **без взгляда на связь с целями**,
# иначе перекрёстная проверка очищенного набора была бы завышена.

# %%
posts = df["vk_wall_posts_collected"].replace(0, np.nan)
df["sex_male"] = (df["vk_sex"] == 2).astype(int)
df["repost_share"] = df["vk_reposts_collected"] / posts
for kind in ("photo", "video", "audio", "link"):
    # Счётчик бывает больше числа постов: это вложения, а не посты
    df[f"{kind}_per_post"] = df[f"vk_posts_with_{kind}_attachments"] / posts
df["has_city"] = df["vk_city_id"].notna().astype(int)
df["has_occupation"] = df["has_occupation_name"]
df["has_university"] = (df["vk_university"] > 0).astype(int)
df["has_relation"] = (df["vk_relation"] > 0).astype(int)
df["profile_block"] = df["len_about"].notna().astype(int)
df["personal_block"] = (df["p_n_personal_fields"] > 0).astype(int)
df["bdate_year_visible"] = df["bdate_year"].notna().astype(int)
df["last_seen_hidden"] = df["vk_last_seen_time"].isna().astype(int)
df["videos_private"] = (df["errcode_videos"] == 30).astype(int)
df["gifts_visible"] = df["vk_counter_gifts"].notna().astype(int)
df["profile_fill"] = sum((df[f"len_{n}"].fillna(0) > 0).astype(int)
                         for n in FREE_TEXT if n != "status")

CLEAN_LOG = ["vk_counter_friends", "vk_counter_followers", "vk_counter_photos", "vk_counter_videos",
             "vk_counter_albums", "vk_counter_audios", "vk_counter_pages", "vk_counter_subscriptions",
             "vk_counter_gifts", "vk_wall_posts_total", "vk_post_avg_likes", "vk_post_avg_comments",
             "vk_post_avg_reposts", "vk_post_avg_views", "vk_post_avg_text_len"]
CLEAN_OTHER = ["sex_male", "repost_share", "photo_per_post", "video_per_post", "audio_per_post",
               "link_per_post", "vk_can_see_audio", "vk_can_see_all_posts",
               "vk_can_write_private_message", "vk_can_post", "has_city", "has_occupation",
               "has_university", "has_relation", "profile_block", "personal_block",
               "bdate_year_visible", "last_seen_hidden", "videos_private", "gifts_visible"]
# Новые признаки из vk_raw, список зафиксирован до проверки (repost_share уже в CLEAN)
NEW_LOG = ["w_posts_per_year", "w_orig_median_text_len"]
# g_admin_share не берём: is_admin считается относительно аккаунта-сборщика
print("Уникальных значений g_admin_share:", df["g_admin_share"].nunique())
NEW_OTHER = ["age", "w_others_share", "profile_fill"]
EXTRA_EXPLORATORY = ["w_night_share", "w_gap_cv", "g_median_log_members", "s_profile_share",
                     "w_n_12m_before", "w_n_12m_after", "w_median_comments"]
CLEAN = CLEAN_LOG + CLEAN_OTHER
CLEAN_NEW = CLEAN + NEW_LOG + NEW_OTHER
print(f"Очищенный набор: {len(CLEAN)}, с новыми: {len(CLEAN_NEW)}")

# %% [markdown]
# ## 4. Связи с чертами: ρ Спирмена, bootstrap, FDR, устойчивость

# %%
N_BOOT = 2000
N_JOBS = 8
SCREEN = list(dict.fromkeys(RAW_FEATURES + CLEAN_NEW + EXTRA_EXPLORATORY))
SCREEN = [c for c in SCREEN if df[c].nunique() > 1]


def spearman_table(feature, data, n_boot, seed):
    x = data[feature].to_numpy(float)
    ok = ~np.isnan(x)
    x, Y = x[ok], data.loc[ok, SHORT].to_numpy(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    rx, rY = stats.rankdata(x), stats.rankdata(Y, axis=0)
    rho = np.array([np.corrcoef(rx, rY[:, j])[0, 1] for j in range(5)])
    boots = np.full((n_boot, 5), np.nan)
    for b in range(n_boot):
        i = rng.integers(n, size=n)
        rxb, rYb = stats.rankdata(x[i]), stats.rankdata(Y[i], axis=0)
        if rxb.std() > 0:
            boots[b] = [np.corrcoef(rxb, rYb[:, j])[0, 1] for j in range(5)]
    t = rho * np.sqrt((n - 2) / np.maximum(1 - rho**2, 1e-12))
    p = 2 * stats.t.sf(np.abs(t), n - 2)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"feature": feature, "trait": SHORT, "n": n, "rho": rho,
                         "ci_low": lo, "ci_high": hi, "p": p})


pairs = pd.concat(Parallel(n_jobs=N_JOBS)(
    delayed(spearman_table)(f, df, N_BOOT, RANDOM_STATE + i) for i, f in enumerate(SCREEN)
), ignore_index=True)
pairs["q"] = multipletests(pairs["p"], method="fdr_bh")[1]

SUBGROUPS = {
    "female": df["sex_male"] == 0, "male": df["sex_male"] == 1,
    "y2022": df["test_year"] == 2022, "y2023plus": df["test_year"] > 2022,
    "no_pilot": df["pilot"] == 0,
}


def sub_rho(feature, trait, mask):
    part = df.loc[mask, [feature, trait]].dropna()
    if len(part) < 30 or part[feature].nunique() < 2:
        return np.nan
    return stats.spearmanr(part[feature], part[trait])[0]


for name, mask in SUBGROUPS.items():
    pairs[f"rho_{name}"] = [sub_rho(f, t, mask) for f, t in zip(pairs["feature"], pairs["trait"])]
sub_cols = [f"rho_{n}" for n in SUBGROUPS]
pairs["same_sign"] = [(np.sign(r[sub_cols].astype(float)) == np.sign(r["rho"])).all() for _, r in pairs.iterrows()]
pairs["stable"] = (pairs["q"] < 0.05) & (pairs["ci_low"] * pairs["ci_high"] > 0) & pairs["same_sign"]
pairs.round(5).to_csv(RESULTS / "spearman_pairs.csv", index=False)

print(f"Пар: {len(pairs)}; p<0.05: {(pairs['p'] < 0.05).sum()}; q<0.05: {(pairs['q'] < 0.05).sum()}; устойчивых: {pairs['stable'].sum()}")
print("По чертам (q<0.05 / устойчивых / макс |ρ|):")
print(pairs.groupby("trait").agg(q05=("q", lambda s: (s < 0.05).sum()), stable=("stable", "sum"),
                                 max_abs=("rho", lambda s: s.abs().max())).round(3))
show = ["feature", "trait", "n", "rho", "ci_low", "ci_high", "q", *sub_cols, "stable"]
print(pairs[pairs["q"] < 0.05].sort_values(["trait", "q"])[show].round(3).to_string())

# %% [markdown]
# ## 5. Нелинейные связи: mutual information и дерево против линейной модели
#
# Для каждой пары: MI с нулевым распределением из перестановок цели (оценка z и
# приближённое p по нормальному закону) и разность CV R² дерева и линейной модели
# на рангах одного признака. Модели обучаются только внутри фолдов.

# %%
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

N_MI_PERM = 200
NL_FEATURES = [c for c in CLEAN_NEW if df[c].nunique() > 2]


def nonlinear_pair(feature, trait, seed):
    part = df[[feature, trait]].dropna()
    X, y = part[[feature]].to_numpy(float), part[trait].to_numpy(float)
    rng = np.random.default_rng(seed)
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=seed)[0]
    null = [mutual_info_regression(X, rng.permutation(y), n_neighbors=5, random_state=seed)[0]
            for _ in range(N_MI_PERM)]
    z = (mi - np.mean(null)) / (np.std(null) + 1e-12)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=40, random_state=seed)
    linear = Pipeline([("rank", QuantileTransformer(n_quantiles=min(200, len(y) // 2), output_distribution="normal")),
                       ("ols", LinearRegression())])
    r2_tree = cross_val_score(tree, X, y, cv=cv, scoring="r2").mean()
    r2_lin = cross_val_score(linear, X, y, cv=cv, scoring="r2").mean()
    return {"feature": feature, "trait": trait, "mi": mi, "mi_z": z, "mi_p": stats.norm.sf(z),
            "r2_tree": r2_tree, "r2_linear": r2_lin, "tree_minus_linear": r2_tree - r2_lin}


RECOMPUTE = os.environ.get("EXPLORE_RECOMPUTE") == "1"
if (RESULTS / "nonlinear_pairs.csv").exists() and not RECOMPUTE:
    nl = pd.read_csv(RESULTS / "nonlinear_pairs.csv")
else:
    nl = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
        delayed(nonlinear_pair)(f, t, RANDOM_STATE + i * 7 + j)
        for i, f in enumerate(NL_FEATURES) for j, t in enumerate(SHORT)
    ))
    nl["mi_q"] = multipletests(nl["mi_p"], method="fdr_bh")[1]
    nl = nl.merge(pairs[["feature", "trait", "rho", "q"]], on=["feature", "trait"], how="left")
    nl["nonlinear_candidate"] = (nl["mi_q"] < 0.05) & (nl["q"] >= 0.05)
    nl.round(5).to_csv(RESULTS / "nonlinear_pairs.csv", index=False)
print(f"Пар {len(nl)}; MI значимо (q<0.05): {(nl['mi_q'] < 0.05).sum()}; "
      f"при этом ранговая связь незначима: {nl['nonlinear_candidate'].sum()}")
print("Дерево лучше линейной модели более чем на 0.005 R²:", int((nl["tree_minus_linear"] > 0.005).sum()))
print(nl.sort_values("mi_z", ascending=False).head(15).round(3).to_string())
print(nl.sort_values("tree_minus_linear", ascending=False).head(8).round(3).to_string())

# %% [markdown]
# ## 6. Шкалы: корреляции и направление
#
# Текстов пунктов в данных нет, поэтому направление можно только косвенно проверить.

# %%
scale_rho = df[SHORT].corr("spearman").round(3)
print(scale_rho)
scale_rho.to_csv(RESULTS / "scale_correlations.csv")
rng = np.random.default_rng(RANDOM_STATE)
rows = []
for i, a in enumerate(SHORT):
    for b in SHORT[i + 1:]:
        boots = [stats.spearmanr(*df.iloc[rng.integers(1000, size=1000)][[a, b]].to_numpy().T)[0] for _ in range(1000)]
        rows.append({"a": a, "b": b, "rho": scale_rho.loc[a, b],
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
print(pd.DataFrame(rows).round(3).to_string())

rows = []
for s in SHORT:
    f, m = df.loc[df["sex_male"] == 0, s], df.loc[df["sex_male"] == 1, s]
    sd = np.sqrt((f.var() + m.var()) / 2)
    age_part = df[["age", s]].dropna()
    rows.append({"trait": s, "mean_female": f.mean(), "mean_male": m.mean(), "d_female_minus_male": (f.mean() - m.mean()) / sd,
                 "p_sex": stats.mannwhitneyu(f, m).pvalue, "rho_age": stats.spearmanr(age_part["age"], age_part[s])[0],
                 "p_age": stats.spearmanr(age_part["age"], age_part[s])[1],
                 "rho_gap": stats.spearmanr(df["gap_years"], df[s])[0], "p_gap": stats.spearmanr(df["gap_years"], df[s])[1]})
demo = pd.DataFrame(rows).round(4)
print(demo.to_string())
demo.to_csv(RESULTS / "trait_demographics.csv", index=False)

print("\nРанние записи с дополнительным вопросом (pilot): n =", int(df["pilot"].sum()),
      "| годы", df.loc[df["pilot"] == 1, "test_year"].value_counts().to_dict(),
      "| всех записей до 2022Q3:", int((test_dt < "2022-07-01").sum()))
for s in SHORT:
    a, b = df.loc[df["pilot"] == 1, s], df.loc[df["pilot"] == 0, s]
    print(f"  {s}: разница средних {a.mean() - b.mean():+.1f}, p={stats.mannwhitneyu(a, b).pvalue:.2g}")

# %% [markdown]
# ## 7. Вмешивающиеся факторы: частный ρ Спирмена
#
# Ранги признака и черты очищаются линейной регрессией от контролей, затем считается
# корреляция остатков. Если признак входит в контроли, эта строка пропускается.
# Активность аккаунта может быть и посредником связи, поэтому её контроль не «истинный» эффект.

# %%
df["ctrl_wall"] = np.log1p(df["vk_wall_posts_total"])
df["ctrl_recency"] = np.log1p((COLLECT_T - df["vk_last_post_date_collected"]).clip(lower=0) / DAY)
df["age2"] = (df["age"] - df["age"].mean()) ** 2
CONTROL_SETS = {
    "sex": ["sex_male"],
    "sex_gap": ["sex_male", "gap_years"],
    "sex_gap_activity": ["sex_male", "gap_years", "ctrl_wall", "ctrl_recency"],
    "sex_gap_age": ["sex_male", "gap_years", "age", "age2"],
    "all": ["sex_male", "gap_years", "ctrl_wall", "ctrl_recency", "age", "age2"],
}


def partial_spearman(feature, trait, controls, data=df):
    cols = [feature, trait, *controls]
    part = data[cols].dropna()
    rx, ry = stats.rankdata(part[feature]), stats.rankdata(part[trait])
    Z = np.column_stack([np.ones(len(part)), part[controls].to_numpy(float)])
    r = np.corrcoef(rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0],
                    ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0])[0, 1]
    dof = len(part) - Z.shape[1] - 1
    p = 2 * stats.t.sf(abs(r) * np.sqrt(dof / max(1 - r**2, 1e-12)), dof)
    return r, p, len(part)


top_pairs = pairs[(pairs["q"] < 0.05) & pairs["feature"].isin(CLEAN_NEW + EXTRA_EXPLORATORY)].copy()
rows = []
age_known = df["age"].notna()
for _, row in top_pairs.iterrows():
    f, t = row["feature"], row["trait"]
    rec = {"feature": f, "trait": t, "rho": row["rho"], "stable": row["stable"]}
    part = df.loc[age_known, [f, t]].dropna()
    rec["rho_age_known"] = stats.spearmanr(part[f], part[t])[0] if part[f].nunique() > 1 else np.nan
    for name, ctrls in CONTROL_SETS.items():
        if f in ctrls or (f == "vk_wall_posts_total" and "ctrl_wall" in ctrls) or (f == "age" and "age" in ctrls):
            rec[name], rec[f"p_{name}"] = np.nan, np.nan
            continue
        rec[name], rec[f"p_{name}"], rec[f"n_{name}"] = partial_spearman(f, t, ctrls)
    rows.append(rec)
partial = pd.DataFrame(rows)
for name in CONTROL_SETS:
    ok = partial[f"p_{name}"].notna()
    partial[f"q_{name}"] = np.nan
    if ok.any():
        partial.loc[ok, f"q_{name}"] = multipletests(partial.loc[ok, f"p_{name}"], method="fdr_bh")[1]
partial.round(4).to_csv(RESULTS / "partial_spearman.csv", index=False)
print(partial[["feature", "trait", "rho", "rho_age_known", *CONTROL_SETS, "q_sex_gap_activity", "q_all", "n_all"]]
      .sort_values(["trait", "rho"]).round(3).to_string())

print("\nЭкстраверсия: связь по терцилям разрыва тест–сбор и в окнах до и после теста")
gap_bin = pd.qcut(df["gap_years"], 3, labels=["малый", "средний", "большой"])
for f in ["vk_counter_friends", "vk_post_avg_likes", "vk_counter_followers"]:
    print(" ", f, {str(g): round(stats.spearmanr(p[f], p["extra"])[0], 3) for g, p in df.groupby(gap_bin, observed=True)})
# Посты до теста видны полностью, только если стена не упёрлась в лимит 200 или дошла до окна
t_test = df["completion_date"] / 1000
window_ok = (df["vk_wall_posts_collected"] < 200) | (df["w_first"] <= t_test - 365 * DAY)
print("  Доля с полным окном 12 мес. до теста:", round(float(window_ok.mean()), 3),
      "| первый собранный пост позже теста:", round(float((df["w_first"] > t_test).mean()), 3))
for s in SHORT:
    p1 = df.loc[window_ok]
    print(f"  {s}: ρ(посты 12м до)={stats.spearmanr(p1['w_n_12m_before'], p1[s])[0]:.3f}, "
          f"ρ(посты 12м после)={stats.spearmanr(p1['w_n_12m_after'], p1[s])[0]:.3f}, "
          f"ρ(лайки до теста)={stats.spearmanr(p1['w_median_likes_before'], p1[s], nan_policy='omit')[0]:.3f}, "
          f"ρ(лайки всего)={stats.spearmanr(p1['w_median_likes'], p1[s])[0]:.3f}")

# %% [markdown]
# ## 8. Информативные пропуски
#
# Закрытых профилей в выборке нет: проверить можно только скрытие отдельных разделов.
# Сбои сбора (`err_friends`, `err_groups`) включены как контрольные: связи быть не должно.

# %%
MISSING_FLAGS = {
    "last_seen_hidden": df["last_seen_hidden"], "videos_private": df["videos_private"],
    "bdate_hidden": 1 - df["has_bdate"], "bdate_year_hidden": 1 - df["bdate_year_visible"],
    "profile_block_absent": 1 - df["profile_block"], "personal_block_absent": 1 - df["personal_block"],
    "city_absent": 1 - df["has_city"], "occupation_absent": 1 - df["has_occupation"],
    "gifts_hidden": 1 - df["gifts_visible"], "groups_counter_absent": df["vk_counter_groups"].isna().astype(int),
    "audio_hidden": 1 - df["vk_can_see_audio"], "posts_restricted": 1 - df["vk_can_see_all_posts"],
    "pm_closed": 1 - df["vk_can_write_private_message"],
    "ctrl_err_friends_rate_limit": df["err_friends"], "ctrl_err_groups_rate_limit": df["err_groups"],
}
rows = []
for name, flag in MISSING_FLAGS.items():
    flag = flag.astype(int)
    for s in SHORT:
        a, b = df.loc[flag == 1, s], df.loc[flag == 0, s]
        rows.append({"flag": name, "share": flag.mean(), "trait": s, "diff": a.mean() - b.mean(),
                     "rho": stats.spearmanr(flag, df[s])[0], "p": stats.mannwhitneyu(a, b).pvalue,
                     "partial_sex_gap": partial_spearman("_flag", s, ["sex_male", "gap_years"],
                                                         data=df.assign(_flag=flag))[0]})
missing = pd.DataFrame(rows)
missing["q"] = multipletests(missing["p"], method="fdr_bh")[1]
missing.round(4).to_csv(RESULTS / "missingness.csv", index=False)
print(missing.pivot(index="flag", columns="trait", values="q").round(3))
print(missing[missing["q"] < 0.1].sort_values("q").round(3).to_string())

# %% [markdown]
# ## 9. Новые признаки из vk_raw

# %%
new_list = ["repost_share", *NEW_LOG, *NEW_OTHER]
print(pairs[pairs["feature"].isin(new_list)].sort_values(["feature", "trait"])
      [["feature", "trait", "n", "rho", "ci_low", "ci_high", "q", "stable"]].round(3).to_string())
print("Связь новых признаков с готовыми (макс. |ρ| и с чем):")
for f in new_list:
    others = df[CLEAN].drop(columns=[f], errors="ignore").corrwith(df[f], method="spearman").abs().dropna().sort_values()
    print(f"  {f}: {others.iloc[-1]:.2f} ({others.index[-1]}), заполнено {df[f].notna().mean():.2f}")

# %% [markdown]
# ## 10. Предсказуемость
#
# Вся предобработка (log1p, импутация с индикаторами пропуска, масштабирование)
# внутри `Pipeline`. Персоны не повторяются, поэтому группировка по людям не нужна.
# Гиперпараметры заданы заранее, а не подбирались на этих же данных.

# %%
N_REPEATS = 5
PERMS = {"ridge": 200, "rf": 100, "hgb": 100}
FEATURE_SETS = {
    "raw": ([], RAW_FEATURES),
    "clean": (CLEAN_LOG, CLEAN_OTHER),
    "clean_new": (CLEAN_LOG + NEW_LOG, CLEAN_OTHER + NEW_OTHER),
    "clean_no_demo": (CLEAN_LOG, [c for c in CLEAN_OTHER if c != "sex_male"]),
    "demo_only": ([], ["sex_male", "age"]),
}


def make_model(name):
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-1, 4, 20))
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33,
                                     n_jobs=1, random_state=RANDOM_STATE)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3,
                                             min_samples_leaf=20, early_stopping=False,
                                             random_state=RANDOM_STATE)
    return DummyRegressor(strategy="mean")


def make_pipeline(model_name, log_cols, other_cols):
    tail = [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    parts = []
    if log_cols:
        log_step = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        parts.append(("log", Pipeline([("log1p", log_step), *tail]), log_cols))
    if other_cols:
        parts.append(("other", Pipeline(tail), other_cols))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", make_model(model_name))])


def cv_scores(model_name, set_name, y, cv, data=df):
    log_cols, other_cols = FEATURE_SETS[set_name]
    pipe = make_pipeline(model_name, log_cols, other_cols)
    return cross_val_score(pipe, data[log_cols + other_cols], y, cv=cv, scoring="r2", n_jobs=1)


jobs = [(m, s, t) for s in FEATURE_SETS for t in SHORT
        for m in (["dummy", "ridge"] if s == "demo_only" else ["dummy", "ridge", "rf", "hgb"])]


def repeated_job(m, s, t):
    cv = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    fold = cv_scores(m, s, df[t].to_numpy(float), cv)
    by_repeat = fold.reshape(N_REPEATS, 5).mean(axis=1)
    return {"model": m, "set": s, "trait": t, "r2": fold.mean(), "sd_repeats": by_repeat.std(),
            "sd_folds": fold.std(), "min_repeat": by_repeat.min(), "max_repeat": by_repeat.max()}


if (RESULTS / "cv_r2.csv").exists() and not RECOMPUTE:
    cv_table = pd.read_csv(RESULTS / "cv_r2.csv")
else:
    cv_table = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(repeated_job)(*j) for j in jobs))
    cv_table.round(4).to_csv(RESULTS / "cv_r2.csv", index=False)
print(cv_table.pivot_table(index=["set", "model"], columns="trait", values="r2")[SHORT].round(3))


def perm_job(m, s, t, seed):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = df[t].to_numpy(float)
    if seed is None:
        return m, s, t, None, cv_scores(m, s, y, cv).mean()
    return m, s, t, seed, cv_scores(m, s, np.random.default_rng(seed).permutation(y), cv).mean()


if (RESULTS / "permutation_tests.csv").exists() and not RECOMPUTE:
    perm_table = pd.read_csv(RESULTS / "permutation_tests.csv")
else:
    # Кандидаты в черту без сигнала дополнительно проверяем без признака пола
    perm_configs = [(m, s, t) for s in ("raw", "clean") for t in SHORT for m in PERMS]
    perm_configs += [(m, "clean_no_demo", t) for t in ("emo", "express") for m in PERMS]
    perm_jobs = [(m, s, t, None) for m, s, t in perm_configs]
    perm_jobs += [(m, s, t, 1000 + i) for m, s, t in perm_configs for i in range(PERMS[m])]
    perm_raw = Parallel(n_jobs=N_JOBS, batch_size=4)(delayed(perm_job)(*j) for j in perm_jobs)
    perm_df = pd.DataFrame(perm_raw, columns=["model", "set", "trait", "seed", "r2"])
    rows = []
    for (m, s, t), g in perm_df.groupby(["model", "set", "trait"]):
        obs = g.loc[g["seed"].isna(), "r2"].iloc[0]
        null = g.loc[g["seed"].notna(), "r2"].to_numpy()
        rows.append({"model": m, "set": s, "trait": t, "r2_obs_single_cv": obs, "null_mean": null.mean(),
                     "null_p95": np.percentile(null, 95), "n_perm": len(null),
                     "p": (1 + (null >= obs).sum()) / (len(null) + 1)})
    perm_table = pd.DataFrame(rows)
    perm_table["q"] = multipletests(perm_table["p"], method="fdr_bh")[1]
    perm_table.round(4).to_csv(RESULTS / "permutation_tests.csv", index=False)
print(perm_table.sort_values(["trait", "set", "model"]).round(3).to_string())

# %% [markdown]
# ## 11. Проверки чувствительности лучшего результата

# %%
cv_rep = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
for t in SHORT:
    no_pilot = df[df["pilot"] == 0]
    r_all = cv_scores("ridge", "clean", df[t].to_numpy(float), cv_rep).mean()
    r_np = cv_scores("ridge", "clean", no_pilot[t].to_numpy(float), cv_rep, data=no_pilot).mean()
    print(f"{t}: Ridge clean R²={r_all:.3f}; без ранних записей {r_np:.3f}")

# Единственный признак: сколько даёт друзья + лайки
FEATURE_SETS["friends_likes"] = (["vk_counter_friends", "vk_post_avg_likes"], [])
print("Экстраверсия, только друзья и лайки:",
      round(cv_scores("ridge", "friends_likes", df["extra"].to_numpy(float), cv_rep).mean(), 3))
# Артефакты сбора: что потеряем, если выбросить только их из исходного набора
ARTEFACTS = ["vk_last_seen_time", "vk_first_post_date_collected", "vk_last_post_date_collected",
             "vk_city_id", "vk_university", "vk_faculty", "vk_occupation_id", "vk_online",
             "vk_counter_online_friends", "vk_photos_kept", "vk_friends_collected", "vk_groups_collected"]
FEATURE_SETS["raw_no_artefacts"] = ([], [c for c in RAW_FEATURES if c not in ARTEFACTS])
for t in SHORT:
    print(f"{t}: raw {cv_scores('ridge', 'raw', df[t].to_numpy(float), cv_rep).mean():.3f}, "
          f"raw без артефактов {cv_scores('ridge', 'raw_no_artefacts', df[t].to_numpy(float), cv_rep).mean():.3f}")

# Сигнал emo: от пола или от поведения в VK?
no_pilot = df[df["pilot"] == 0]
print("emo без ранних записей, Ridge без пола:",
      round(cv_scores("ridge", "clean_no_demo", no_pilot["emo"].to_numpy(float), cv_rep, data=no_pilot).mean(), 3))
sex_gap = ["sex_male", "gap_years"]
print("Связи express при контроле экстраверсии (ρ → частный ρ):")
for f in ["vk_post_avg_likes", "vk_counter_friends", "audio_per_post", "repost_share", "vk_counter_photos"]:
    print(f"  {f}: {stats.spearmanr(df[f], df['express'], nan_policy='omit')[0]:.3f} → "
          f"{partial_spearman(f, 'express', ['extra'])[0]:.3f}")
print("Связи экстраверсии при контроле express (ρ → частный ρ):")
for f in ["vk_post_avg_likes", "vk_counter_friends", "vk_counter_followers"]:
    print(f"  {f}: {stats.spearmanr(df[f], df['extra'], nan_policy='omit')[0]:.3f} → "
          f"{partial_spearman(f, 'extra', ['express'])[0]:.3f}")
# Коды вуза «коррелируют» из-за нуля-заглушки: среди настоящих кодов связь должна исчезнуть
print("Коды вуза и факультета только среди ненулевых значений:")
for f in ["vk_faculty", "vk_university"]:
    part = df[df[f] > 0]
    print(f"  {f} (n={len(part)}):", {s: round(stats.spearmanr(part[f], part[s])[0], 3) for s in SHORT})
    print(f"    p:", {s: round(stats.spearmanr(part[f], part[s])[1], 3) for s in SHORT})
print("Разрыв и control без ранних записей: ρ =",
      round(stats.spearmanr(no_pilot["gap_years"], no_pilot["control"])[0], 3))
print("Ночные посты и control с контролем возраста:", [round(v, 3) for v in partial_spearman("w_night_share", "control", ["sex_male", "gap_years", "age", "age2"])])
print("Готово. Агрегаты сохранены в", RESULTS)
