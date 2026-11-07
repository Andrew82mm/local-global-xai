"""Реальные данные: загрузка очищенной таблицы из кеша и построение 34 признаков.

Источник: `data/exploration_table.csv.gz`, его создаёт `notebooks/03_data_exploration.py`.
Сырой `data/dataset_1000.jsonl` здесь не читается.

Персональные данные: функции возвращают таблицы с данными людей. Выводить на экран
и сохранять можно только агрегаты.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "exploration_table.csv.gz"

TARGETS = {
    "target_extraversion_introversion": "extra",
    "target_attachment_detachment": "attach",
    "target_self_control_impulsivity": "control",
    "target_emotional_stability_instability": "emo",
    "target_expressiveness_practicality": "express",
}

# Счётчики с сильной правой асимметрией: log1p внутри Pipeline
FEATURES_LOG = [
    "vk_counter_friends", "vk_counter_followers", "vk_counter_photos", "vk_counter_videos",
    "vk_counter_albums", "vk_counter_audios", "vk_counter_pages", "vk_counter_subscriptions",
    "vk_counter_gifts", "vk_wall_posts_total", "vk_post_avg_likes", "vk_post_avg_comments",
    "vk_post_avg_reposts", "vk_post_avg_views", "vk_post_avg_text_len",
]
# Доли (непрерывные) и флаги 0/1. sex_male убран из основного эксперимента
FEATURES_RATIO = ["repost_share", "photo_per_post", "video_per_post", "audio_per_post", "link_per_post"]
FEATURES_FLAG = [
    "vk_can_see_audio", "vk_can_see_all_posts", "vk_can_write_private_message", "vk_can_post",
    "has_city", "has_occupation", "has_university", "has_relation", "profile_block",
    "personal_block", "bdate_year_visible", "last_seen_hidden", "videos_private", "gifts_visible",
]
FEATURES_OTHER = FEATURES_RATIO + FEATURES_FLAG
FEATURES = FEATURES_LOG + FEATURES_OTHER
# Для PDP и ALE берём только непрерывные признаки: у флага кривая из двух точек
CONTINUOUS = FEATURES_LOG + FEATURES_RATIO

# Колонки кеша, которые нельзя использовать как признаки ни при каких условиях:
# raw_* продублированы из test_raw_result_json (точные значения целей), test_* и
# completion_date (миллисекунды) служебные поля теста.
FORBIDDEN_PREFIXES = ("raw_", "test_", "target_")
FORBIDDEN = {"completion_date", "g_admin_share", "person", "vk_id_rank"}


def load_table(path=CACHE_PATH):
    """Читает кеш и переименовывает цели в короткие имена (extra, emo, ...)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Нет кеша {path}: сначала запустите notebooks/03_data_exploration.py")
    return pd.read_csv(path).rename(columns=TARGETS)


def build_features(df):
    """34 признака очищенного набора без пола, до log1p, импутации и масштабирования.

    Повторяет ячейку «Наборы признаков» из ноутбука 03. Пропуски остаются как NaN:
    их заполняет Pipeline внутри фолда.
    """
    out = pd.DataFrame(index=df.index)
    posts = df["vk_wall_posts_collected"].replace(0, np.nan)
    for col in FEATURES_LOG:
        out[col] = df[col]
    out["repost_share"] = df["vk_reposts_collected"] / posts
    for kind in ("photo", "video", "audio", "link"):
        # Счётчик бывает больше числа постов: это вложения, а не посты
        out[f"{kind}_per_post"] = df[f"vk_posts_with_{kind}_attachments"] / posts
    for col in ("vk_can_see_audio", "vk_can_see_all_posts", "vk_can_write_private_message", "vk_can_post"):
        out[col] = df[col]
    out["has_city"] = df["vk_city_id"].notna().astype(int)
    out["has_occupation"] = df["has_occupation_name"]
    out["has_university"] = (df["vk_university"] > 0).astype(int)
    out["has_relation"] = (df["vk_relation"] > 0).astype(int)
    out["profile_block"] = df["len_about"].notna().astype(int)
    out["personal_block"] = (df["p_n_personal_fields"] > 0).astype(int)
    out["bdate_year_visible"] = df["bdate_year"].notna().astype(int)
    out["last_seen_hidden"] = df["vk_last_seen_time"].isna().astype(int)
    out["videos_private"] = (df["errcode_videos"] == 30).astype(int)
    out["gifts_visible"] = df["vk_counter_gifts"].notna().astype(int)
    out = out[FEATURES].astype(float)
    _check_no_leak(out.columns)
    return out


def _check_no_leak(columns):
    bad = [c for c in columns if c.startswith(FORBIDDEN_PREFIXES) or c in FORBIDDEN]
    if bad:
        raise ValueError(f"Запрещённые колонки среди признаков: {bad}")


def load_real(target, path=CACHE_PATH):
    """Признаки X (34 колонки) и цель y для одной черты: 'extra', 'emo', ..."""
    df = load_table(path)
    if target not in TARGETS.values():
        raise ValueError(f"Неизвестная черта {target!r}")
    return build_features(df), df[target].to_numpy(float)
