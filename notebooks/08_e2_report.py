# %% [markdown]
# # 08. E2: рисунки отчёта и аддитивный контроль
#
# Ноутбук строит рисунки для отчёта 2 и сравнивает PDP и ALE при сумме и произведении:
# - что выучила модель: PI, PDP и ALE на главных признаках;
# - формы зависимостей модели против истинной функции E2;
# - SHAP summary и heatmap локальных объяснений;
# - объекты, у которых локальное объяснение сильнее всего отличается от среднего.
#
# Берётся один уровень сигнала R² = 0.2: на нём формы кривых видны лучше всего.
# Для PI и кривых — среднее по 10 повторам, для summary/heatmap — один
# представительный повтор (seed 0), чтобы точки соответствовали конкретным объектам.
#
# **Данные анонимизированы.** В файлах и на рисунках нет имён, id и сырых профилей:
# только номера строк тестовой выборки, SHAP-вклады и агрегаты.
#
# Запуск: `uv run python notebooks/08_e2_report.py` (около 3 минут).
# Быстрая проба: `E2_FIG_SMOKE=1 uv run python notebooks/08_e2_report.py`.
# Для сравнений нужны агрегаты ноутбуков 06 и 07.
# Локальный кеш: `results/e2_fig/runs.pkl`, пересчёт: `E2_FIG_RECOMPUTE=1`.

# %%
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.data import CONTINUOUS, FEATURES_LOG, FEATURES_OTHER, load_real  # noqa: E402
from xai_consistency.semisynthetic import (  # noqa: E402
    FoldTruth, Term, Truth, TWIN, fold_affine, make_target, to_explain_space,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)

SMOKE = os.environ.get("E2_FIG_SMOKE") == "1"
RECOMPUTE = os.environ.get("E2_FIG_RECOMPUTE") == "1"
N_SEEDS = 1 if SMOKE else 10
R2 = 0.2
SEED_HEATMAP = 0
TEST_SIZE = 0.2
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CACHE_DIR = RESULTS / ("e2_fig_smoke" if SMOKE else "e2_fig")
FIGURES.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Цвета те же, что в ноутбуках 06–10.
C = {"pi": "#2a78d6", "shap": "#eb6834", "lime": "#1baf7a", "pdp": "#eda100", "ale": "#e87ba4",
     "rf": "#2a78d6", "hgb": "#eb6834", "ridge": "#1baf7a",
     "true": "#0b0b0b", "null": "#9a9893", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
NAME = {"pi": "PI", "shap": "SHAP", "lime": "LIME", "pdp": "PDP", "ale": "ALE",
        "rf": "RF", "hgb": "HGB", "ridge": "Ridge"}

FEAT_RU = {
    "vk_counter_friends": "друзья",
    "vk_counter_followers": "подписчики",
    "vk_counter_photos": "фото",
    "vk_counter_videos": "видео",
    "vk_counter_albums": "альбомы",
    "vk_counter_audios": "аудио",
    "vk_counter_pages": "сообщества",
    "vk_counter_subscriptions": "подписки",
    "vk_counter_gifts": "подарки",
    "vk_wall_posts_total": "посты",
    "vk_post_avg_likes": "лайки",
    "vk_post_avg_comments": "комментарии",
    "vk_post_avg_reposts": "репосты",
    "vk_post_avg_views": "просмотры",
    "vk_post_avg_text_len": "длина текста",
    "repost_share": "доля репостов",
    "photo_per_post": "фото на пост",
    "video_per_post": "видео на пост",
    "audio_per_post": "аудио на пост",
    "link_per_post": "ссылки на пост",
    "vk_can_see_audio": "аудио открыто",
    "vk_can_see_all_posts": "все посты видны",
    "vk_can_write_private_message": "можно писать",
    "vk_can_post": "можно на стене",
    "has_city": "указан город",
    "has_occupation": "указана работа",
    "has_university": "указан вуз",
    "has_relation": "семейное положение",
    "profile_block": "профиль заполнен",
    "personal_block": "личные поля",
    "bdate_year_visible": "год рождения виден",
    "last_seen_hidden": "был(а) скрыт",
    "videos_private": "видео скрыты",
    "gifts_visible": "подарки видны",
}


def flabel(f):
    return FEAT_RU.get(f, f)


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
    "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
SAVED = []


def save(fig, name, title):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    SAVED.append((name, title))
    print(f"{name}: {title}")


# %% [markdown]
# ## 1. Данные и истинная функция
#
# Признаки настоящие, цель синтетическая: y = √R²·f + шум, как в E2.
# Истина задана в пространстве после log1p и стандартизации.

# %%
X, _ = load_real("extra")
T_FULL = to_explain_space(X)
TRUTH = Truth(T_FULL)
RELEVANT = list(TRUTH.relevant)
INTER = list(TRUTH.terms[-1].features)
TWIN_EXCLUDED = TWIN[1]
CURVE_FEATURES = [f for f in RELEVANT if f in CONTINUOUS]
print("Истинные признаки:", RELEVANT)
print("Взаимодействие:", INTER)
print("Двойник вне f:", TWIN_EXCLUDED)
print("Кривые:", CURVE_FEATURES)


# %% [markdown]
# ## 2. Прогоны
#
# Для каждого seed обучаются три модели. Для Ridge достаточно PI: его SHAP, PDP и ALE
# сводятся к коэффициентам. Для RF и HGB сохраняются PI, SHAP, PDP и ALE.
# Дополнительно для каждого объекта считается отклонение вектора |SHAP| от среднего
# и согласие с истинным SHAP.

# %%
def center_curve(curve):
    """Центрирование кривой по числу объектов в интервалах."""
    w = curve["count"].to_numpy(float)
    v = curve["value"].to_numpy(float)
    ok = np.isfinite(v) & (w > 0)
    if ok.sum() == 0:
        return v
    return v - np.sum(v[ok] * w[ok]) / np.sum(w[ok])


def compute_tables():
    pi_rows, shap_rows, curve_rows, local_rows, top_rows, dev_rows = [], [], [], [], [], []
    shap_store = {}
    t0 = time.perf_counter()

    for seed in range(N_SEEDS):
        y, _ = make_target(TRUTH, T_FULL, R2, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed)

        res_by_model = {}
        for model in ("ridge", "rf", "hgb"):
            methods = ("pi",) if model == "ridge" else ("pi", "pdp", "ale", "shap")
            res = E.fit_and_explain(
                model, X_train, y_train, X_test, y_test,
                log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                curve_features=CONTINUOUS, seed=seed,
                methods=methods, splits=("test",),
            )
            res_by_model[model] = res
            for f, v in res[("pi", "test")].importance.items():
                pi_rows.append({"seed": seed, "model": model, "feature": f, "importance": float(v)})
            if model != "ridge":
                for f, v in res[("shap", "test")].importance.items():
                    shap_rows.append({"seed": seed, "model": model, "feature": f, "importance": float(v)})

        for model in ("hgb", "rf"):
            res = res_by_model[model]
            ft = FoldTruth(TRUTH, res.pipe, res.features, R2)
            aff = fold_affine(res.pipe)
            Z_expl = res.samples["test"]
            phi = ft.true_shap(Z_expl, res.background).abs()

            # --- кривые ---
            for f in CURVE_FEATURES:
                grids = {f: res.grids[f]}
                pdp_true = E.partial_dependence(ft, res.Z_test, grids).curves[f]
                ale_true = E.accumulated_local_effects(ft, res.Z_test, grids).curves[f]
                curves = {
                    "pdp": res[("pdp", "test")].curves[f],
                    "ale": res[("ale", "test")].curves[f],
                    "pdp_true": pdp_true,
                    "ale_true": ale_true,
                }
                if f in RELEVANT and f not in INTER:
                    curves["main_true"] = pdp_true.assign(
                        value=ft.main_curve(f, pdp_true["x"].to_numpy()))
                m, s = aff[f]
                for method, c in curves.items():
                    x_t = c["x"].to_numpy(float) * s + m
                    vals = center_curve(c)
                    for x, v in zip(x_t, vals):
                        curve_rows.append({"seed": seed, "model": model, "method": method,
                                           "feature": f, "x_t": float(x), "value": float(v)})

            # --- локальные объяснения ---
            local_signed = res[("shap", "test")].local
            sh = local_signed.abs()
            mean_vec = sh.mean()
            d = np.sqrt(((sh - mean_vec) ** 2).sum(axis=1))
            rho = np.array([
                M.rank_agreement(sh.iloc[i], phi.iloc[i], RELEVANT)["spearman"]
                for i in range(len(sh))
            ], float)
            top3 = np.array([
                len(set(M.top_k(sh.iloc[i], 3)) & set(M.top_k(phi.iloc[i], 3))) / 3
                for i in range(len(sh))
            ], float)
            for i in range(len(sh)):
                local_rows.append({"seed": seed, "model": model, "object": i + 1,
                                   "deviation": float(d.iloc[i]), "rho_true6": rho[i],
                                   "top3_true": top3[i]})
            # Вклад признаков в отклонение: топ-5% объектов против остальных.
            cut = np.quantile(d, 0.95)
            top_mask = d >= cut
            for f in sh.columns:
                dev = (sh[f] - mean_vec[f]).abs()
                dev_rows.append({"seed": seed, "model": model, "feature": f, "group": "top5",
                                 "deviation": float(dev[top_mask].mean()) if top_mask.any() else np.nan})
                dev_rows.append({"seed": seed, "model": model, "feature": f, "group": "rest",
                                 "deviation": float(dev[~top_mask].mean()) if (~top_mask).any() else np.nan})

            if seed == SEED_HEATMAP and model == "hgb":
                shap_store = {"shap": local_signed.copy(), "Z": Z_expl.copy(), "phi": phi.copy(),
                              "deviation": d.copy(), "rho": rho.copy(), "top3": top3.copy(),
                              "mean_vec": mean_vec.copy()}
                order = d.sort_values(ascending=False).index[:5]
                for i in order:
                    feats = (sh.iloc[i] - mean_vec).abs().sort_values(ascending=False).index[:3]
                    top_rows.append({
                        "object": i + 1,
                        "deviation": float(d.iloc[i]),
                        "rho_true6": float(rho[i]),
                        "top3_true": float(top3[i]),
                        "features": "; ".join(flabel(f) for f in feats),
                        "shap": "; ".join(f"{sh.iloc[i][f]:.2f}" for f in feats),
                        "true_shap": "; ".join(f"{phi.iloc[i][f]:.2f}" for f in feats),
                    })

        print(f"seed {seed}: {time.perf_counter() - t0:.1f} с")

    return (pd.DataFrame(pi_rows), pd.DataFrame(shap_rows), pd.DataFrame(curve_rows),
            pd.DataFrame(local_rows), pd.DataFrame(top_rows), pd.DataFrame(dev_rows), shap_store)


# %%
# Кеш хранит все таблицы вместе, чтобы не обучать модели повторно.
if (CACHE_DIR / "runs.pkl").exists() and not RECOMPUTE:
    import pickle
    with open(CACHE_DIR / "runs.pkl", "rb") as fh:
        pi_tab, shap_tab, curve_tab, local_tab, top_tab, dev_tab, shap_store = pickle.load(fh)
else:
    pi_tab, shap_tab, curve_tab, local_tab, top_tab, dev_tab, shap_store = compute_tables()
    import pickle
    with open(CACHE_DIR / "runs.pkl", "wb") as fh:
        pickle.dump((pi_tab, shap_tab, curve_tab, local_tab, top_tab, dev_tab, shap_store), fh)

print("PI:", pi_tab.shape, "| SHAP importance:", shap_tab.shape, "| curves:", curve_tab.shape)
print("local:", local_tab.shape, "| top objects:", top_tab.shape, "| deviations:", dev_tab.shape)

# %% [markdown]
# ## 3. PI: какие признаки использует модель
#
# Среднее падение R² при перемешивании по 10 повторам. Истинные признаки выделены
# цветом, двойник — отдельным цветом. У PI нет знака, поэтому направление эффекта
# показывают SHAP и кривые ниже.

# %%
pi_mean = pi_tab.groupby(["model", "feature"])["importance"].mean().unstack("model")
pi_mean = pi_mean.reindex(columns=["ridge", "rf", "hgb"])
# Порядок: топ-12 по средней важности, плюс все истинные признаки и двойник.
order = pi_mean.mean(axis=1).sort_values(ascending=False).index.tolist()[:12]
for f in RELEVANT + [TWIN_EXCLUDED]:
    if f not in order:
        order.append(f)
order = order[::-1]  # сверху — самые важные

fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
for ax, model in zip(axes, ("ridge", "rf", "hgb")):
    vals = pi_mean[model].reindex(order)
    y = np.arange(len(order))
    colors = [C["lime"] if f == TWIN_EXCLUDED else C["shap"] if f in RELEVANT else C["null"] for f in order]
    ax.barh(y, vals.values, color=colors, height=0.72)
    ax.axvline(0, color=C["ink2"], lw=0.8)
    ax.set_yticks(y, [flabel(f) for f in order])
    ax.set_title(NAME[model], loc="left")
    ax.set_xlabel("падение R² при перемешивании")
    ax.grid(axis="y", visible=False)
handles = [Patch(color=C["shap"], label="истинный признак"),
           Patch(color=C["lime"], label="двойник вне f"),
           Patch(color=C["null"], label="шумовой признак")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("PI: какие признаки использует модель (R² = 0.2, среднее по 10 повторам)",
             x=0.01, y=1.03, ha="left", fontsize=10.5)
fig.text(0.01, -0.12, "Столбики — среднее падение R² на тесте. Отрицательная PI означает рост тестового R² после перемешивания. "
         "У Ridge вклад делится между audios и его двойником can_see_audio.",
         fontsize=7.5, color=C["ink2"])
fig.tight_layout()
save(fig, "fig11_e2_importance", "PI: важность признаков в E2")

# %% [markdown]
# ## 4. PDP и ALE: формы зависимостей модели против истины
#
# Кривые центрированы. Для отдельных эффектов по X — исходное значение признака
# (symlog). Для взаимодействия по X — стандартизованный log1p, как в формуле f.

# %%
# Общая сетка по T (после log1p) для усреднения кривых по повторам.
grids_t = {}
for f in CURVE_FEATURES:
    vals = T_FULL[f].dropna().to_numpy(float)
    grids_t[f] = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 25)))

avg_rows = []
for (model, method, feature), g in curve_tab.groupby(["model", "method", "feature"]):
    grid = grids_t[feature]
    curves = []
    for _, gg in g.groupby("seed"):
        x = gg["x_t"].to_numpy(float)
        v = gg["value"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(v)
        if ok.sum() >= 2:
            curves.append(np.interp(grid, x[ok], v[ok], left=np.nan, right=np.nan))
    if curves:
        mean = np.nanmean(np.vstack(curves), axis=0)
        for x, v in zip(grid, mean):
            avg_rows.append({"model": model, "method": method, "feature": feature, "x_t": x, "value": v})
avg_curves = pd.DataFrame(avg_rows)
avg_curves["x_raw"] = np.expm1(avg_curves["x_t"])

fig, axes = plt.subplots(1, 3, figsize=(8.3, 2.7), layout="constrained")
main_features = ("vk_counter_friends", "vk_post_avg_likes", "vk_counter_audios")
for ax, f in zip(axes, main_features):
    sub = avg_curves[(avg_curves["model"] == "hgb") & (avg_curves["feature"] == f)]
    for method, color, ls, lab in (
        ("pdp", C["pdp"], "-", "PDP модели"),
        ("ale", C["ale"], "--", "ALE модели"),
        ("main_true", C["true"], "-", "Заданный эффект"),
    ):
        d = sub[sub["method"] == method]
        ax.plot(d["x_raw"], d["value"], color=color, ls=ls, label=lab)
    ax.axhline(0, color=C["ink2"], lw=0.7, alpha=0.6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.set_title(flabel(f))
axes[0].set_ylabel("Центрированный эффект")
axes[1].set_xlabel("Исходное значение признака (symlog)")
axes[0].legend(fontsize=8, loc="upper left")
save(fig, "fig12_e2_curves", "PDP и ALE модели для трёх отдельных эффектов")

# %% [markdown]
# ## 4.1. Кривые взаимодействия для заданного сигнала и прогноза

# %%
fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.8), sharex="row", sharey=True,
                         layout="constrained")
for row, f in enumerate(INTER):
    sub = avg_curves[(avg_curves["model"] == "hgb") & (avg_curves["feature"] == f)]
    for col, suffix in enumerate(("_true", "")):
        ax = axes[row, col]
        for method, color, ls in (("pdp", "#17609c", "-"), ("ale", "#b04918", "--")):
            d = sub[sub["method"] == method + suffix]
            z = (d["x_t"] - TRUTH.mu[f]) / TRUTH.sd[f]
            ax.plot(z, d["value"], color=color, ls=ls, label=method.upper())
        ax.axhline(0, color=C["ink2"], lw=0.7, alpha=0.6)
        ax.set_xlabel(f"{flabel(f).capitalize()}, z")
        ax.tick_params(labelbottom=True)
axes[0, 0].set_title("Заданный сигнал")
axes[0, 1].set_title("Прогноз HGB")
axes[0, 0].legend(loc="best")
fig.supylabel("Центрированный эффект")
save(fig, "fig15_e2_interaction_curves", "PDP и ALE для фото и постов")

# %% [markdown]
# ## 4.2. Аддитивный контроль: фото + посты
#
# Сравниваем две заданные функции без шума и обучения. Используем 30 тестовых
# подвыборок E2 и ту же квантильную сетку внутри каждой подвыборки. Считаем
# в координатах после log1p: аффинная стандартизация фолда не меняет эти кривые.

# %%
z_pair = (T_FULL[INTER] - TRUTH.mu[INTER]) / TRUTH.sd[INTER]
# S(z_a + z_b) = (z_a + z_b) / SD(z_a + z_b), так как каждое z центрировано.
# Два линейных Term в сумме воспроизводят одно нормированное слагаемое.
sum_weight = TRUTH.terms[-1].weight / z_pair.sum(axis=1).std(ddof=0)
additive = Truth(T_FULL, terms=TRUTH.terms[:-1] + tuple(
    Term("main", (f,), weight=sum_weight) for f in INTER
))
control_rows = []
for seed in range(30):
    _, T_test = train_test_split(T_FULL[RELEVANT], test_size=TEST_SIZE, random_state=seed)
    grids = E.make_grids(T_test, INTER)
    for variant, truth in (("произведение", TRUTH), ("сумма", additive)):
        signal = SimpleNamespace(predict=lambda arr, truth=truth: np.sqrt(R2) * truth.f(
            pd.DataFrame(arr, columns=RELEVANT)
        ))
        pdp = E.partial_dependence(signal, T_test, grids).curves
        ale = E.accumulated_local_effects(signal, T_test, grids).curves
        control_rows.append(M.compare_curves(pdp, ale).assign(variant=variant, seed=seed))

control = pd.concat(control_rows)
print("\nАддитивный контроль, средние по фото, постам и 30 разбиениям:")
print(control.groupby("variant")[["ccc", "amp_a", "amp_b"]].mean().round(6))

# %% [markdown]
# ## 5. SHAP summary и heatmap
#
# Один представительный повтор (seed 0, HGB, 200 тестовых объектов). Summary
# показывает направление и разброс локальных вкладов; heatmap — те же вклады
# построчно, объекты отсортированы по вкладу самого важного признака.

# %%
sh_signed = shap_store["shap"]
sh_abs = sh_signed.abs()
Z0 = shap_store["Z"]
phi = shap_store["phi"]
top_feats = sh_abs.mean().sort_values(ascending=False).index[:12].tolist()

# --- summary ---
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
feats_plot = top_feats[::-1]
rng = np.random.default_rng(0)
for i, f in enumerate(feats_plot):
    x = sh_signed[f].to_numpy(float)
    c = Z0[f].to_numpy(float)
    c = (c - np.nanmin(c)) / (np.nanmax(c) - np.nanmin(c)) if np.nanmax(c) > np.nanmin(c) else np.full_like(c, 0.5)
    y = i + rng.uniform(-0.34, 0.34, len(x))
    ax.scatter(x, y, c=c, cmap="coolwarm", vmin=0, vmax=1, s=10, alpha=0.75, linewidths=0)
ax.axvline(0, color=C["ink2"], lw=0.8)
ax.set_yticks(range(len(feats_plot)), [flabel(f) for f in feats_plot])
ax.set_xlabel("SHAP-вклад в прогноз объекта")
ax.set_title("а) SHAP summary: 200 объектов", loc="left")
ax.grid(axis="y", visible=False)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cb.set_ticks([0, 1])
cb.set_ticklabels(["меньше", "больше"])
cb.set_label("значение признака", fontsize=8)

# --- heatmap ---
ax = axes[1]
sort_feat = top_feats[0]
obj_order = sh_signed[sort_feat].sort_values(ascending=False).index
mat = sh_signed.loc[obj_order, top_feats].T.to_numpy(float)
vmax = np.nanpercentile(np.abs(mat), 95)
im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_yticks(range(len(top_feats)), [flabel(f) for f in top_feats])
ax.set_xticks([])
ax.set_xlabel(f"объекты (сортировка по вкладу «{flabel(sort_feat)}»)")
ax.set_title("б) SHAP heatmap: те же 200 объектов", loc="left")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("SHAP, единицы прогноза", fontsize=8)
fig.suptitle("Локальные объяснения HGB: направление, разброс и отдельные объекты (R² = 0.2, seed 0)",
             x=0.01, y=1.03, ha="left", fontsize=10.5)
fig.text(0.01, -0.04, "Каждая точка summary — объект; цвет — значение признака. "
         "В heatmap строки — признаки, столбцы — объекты. Данные анонимизированы: показаны только номера строк.",
         fontsize=7.5, color=C["ink2"])
fig.tight_layout()
save(fig, "fig13_e2_shap_summary", "SHAP summary и heatmap (E2)")

# %% [markdown]
# ## 6. Необычные объекты: у кого локальное объяснение сильнее всего отличается от среднего
#
# Для каждого объекта считается L2-расстояние между вектором |SHAP| и средним вектором
# по выборке. Затем проверяется, связана ли необычность с ошибкой относительно истинного
# SHAP: ρ по шести истинным признакам и совпадение top-3. Отдельно — какие признаки
# дают основной вклад в отклонение у топ-5% объектов.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))

ax = axes[0]
ax.hist(local_tab["deviation"], bins=40, color=C["shap"], alpha=0.75, edgecolor="white")
cut = local_tab["deviation"].quantile(0.95)
ax.axvline(cut, color=C["ink"], ls="--", lw=1.2)
ax.text(cut, ax.get_ylim()[1] * 0.92, "  топ-5%", fontsize=8, color=C["ink"])
ax.set_xlabel("отклонение |SHAP| от среднего")
ax.set_ylabel("объектов")
ax.set_title("а) Есть объекты, непохожие на среднее", loc="left")

ax = axes[1]
ax.scatter(local_tab["deviation"], local_tab["rho_true6"], s=8, alpha=0.2, color=C["shap"], linewidths=0)
r = stats.spearmanr(local_tab["deviation"], local_tab["rho_true6"], nan_policy="omit").statistic
ax.axhline(0, color=C["ink2"], lw=0.8)
ax.set_xlabel("отклонение |SHAP| от среднего")
ax.set_ylabel("ρ с истинным SHAP по 6 признакам")
ax.set_title("б) Необычность и верность истине", loc="left")
ax.text(0.02, 0.95, f"ρ Спирмена = {r:.2f}", transform=ax.transAxes, fontsize=8,
        va="top", color=C["ink2"])

ax = axes[2]
dev = dev_tab.groupby(["model", "feature", "group"])["deviation"].mean().unstack("group")
dev = dev.loc["hgb"]
top_dev = (dev["top5"] - dev["rest"]).sort_values(ascending=False).index[:10][::-1]
y = np.arange(len(top_dev))
ax.barh(y - 0.18, dev.loc[top_dev, "top5"], height=0.34, color=C["shap"], label="топ-5%")
ax.barh(y + 0.18, dev.loc[top_dev, "rest"], height=0.34, color=C["null"], label="остальные")
ax.set_yticks(y, [flabel(f) for f in top_dev])
ax.set_xlabel("среднее |SHAP − среднее|")
ax.set_title("в) Чем необычные объекты отличаются", loc="left")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)

fig.suptitle("Локальные выбросы: какие объекты объясняются не как все",
             x=0.01, y=1.03, ha="left", fontsize=10.5)
fig.text(0.01, -0.05, "HGB и RF, 10 повторов. Топ-5% — объекты с наибольшим отклонением вектора |SHAP| "
         "от среднего по выборке. На панели (в) — HGB; для RF порядок похож.",
         fontsize=7.5, color=C["ink2"])
fig.tight_layout()
save(fig, "fig14_e2_local_outliers", "Необычные объекты в локальных объяснениях (E2)")

# %% [markdown]
# ## 7. Пять самых необычных объектов (seed 0, HGB)
#
# Номера объектов — это номера строк тестовой выборки, не идентификаторы людей.

# %%
print(top_tab.to_string(index=False))
print("\nСвязь необычности и верности истине:")
print(local_tab.groupby("model")[["deviation", "rho_true6", "top3_true"]].mean().round(3))
print("\nКорреляция отклонения и ρ с истинным SHAP:")
for model, g in local_tab.groupby("model"):
    print(model, round(stats.spearmanr(g["deviation"], g["rho_true6"], nan_policy="omit").statistic, 3))

# %% [markdown]
# ## 8. Сравнения для отчёта из агрегатов E1 и E2
#
# Таблицы создают ноутбуки 06 и 07. Имена сценариев, в том числе `null`,
# читаются как строки.

# %%
def read(name):
    return pd.read_csv(RESULTS / f"{name}.csv", keep_default_na=False, na_values=[""], low_memory=False)


MARK = {"pi": "o", "shap": "s", "lime": "^", "pdp": "D", "ale": "v", "rf": "o", "hgb": "s"}

def r2_axis(ax, levels):
    """Уровни R² на равномерной оси (они не равноотстоят)."""
    ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
    ax.set_xlabel("Уровень сигнала (отдельные условия)")


TEST = ["pi_test", "shap_test", "lime_test", "pdp_test", "ale_test"]


def split_pairs(p):
    """Средний ρ внутри SHAP/LIME/PDP/ALE и PI с остальными по каждому прогону."""
    p = p[p["a"].isin(TEST) & p["b"].isin(TEST)]
    is_pi = (p["a"] == "pi_test") | (p["b"] == "pi_test")
    return p.assign(group=np.where(is_pi, "pi", "within"))

e1p = split_pairs(read("e1_pairs"))
e2p = split_pairs(read("e2_pairs").query("set == 'continuous20'"))
E1_R2 = [0.0, 0.02, 0.08, 0.2, 0.5]
E2_R2 = [0.0, 0.02, 0.08, 0.2]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, gridspec_kw={"width_ratios": [5, 4]})
for ax, p, levels, title in ((axes[0], e1p, E1_R2, "E1: синтетика"), (axes[1], e2p, E2_R2, "E2: полусинтетика")):
    m = p[p["model"].isin(["rf", "hgb"])].groupby(["group", "model", "r2"])["spearman"].mean()
    for grp, col, lab in (("within", C["shap"], "внутри SHAP, LIME, PDP, ALE"), ("pi", C["pi"], "PI с остальными")):
        for mod, ls in (("rf", "-"), ("hgb", "--")):
            y = m.loc[(grp, mod)].reindex(levels)
            ax.plot(range(len(levels)), y.values, ls=ls, color=col, marker=MARK[mod], markersize=6,
                    label=f"{lab}, {NAME[mod]}")
    r2_axis(ax, levels)
    ax.set_title(title, loc="left")
    ax.axhline(0, color=C["ink2"], lw=0.8)
axes[0].set_ylabel("средний ρ Спирмена между методами")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Согласие SHAP, LIME, PDP и ALE высоко уже при нулевом сигнале",
             x=0.01, y=1.03, ha="left", fontsize=10.5)
save(fig, "fig01a_agreement_e12", "Согласие методов против R² (E1, E2), без реальных данных")

# %% [markdown]
# ### Полнота и знаки в E1 и E2

# %%
e1m = read("e1_methods")
e1m = e1m[e1m["method"].isin(TEST) & e1m["scenario"].isin(["independent", "correlated ρ=0.5"])]
e2m = read("e2_methods")
e2m = e2m[e2m["method"].isin(TEST)]
fig, axes = plt.subplots(2, 2, figsize=(8.5, 6), sharex="col")
recall_e2_methods = ("pi_test", "shap_test", "lime_test")
for col, (d, levels, k, rnd, tag) in enumerate(((e1m, E1_R2[1:], 4, 0.4, "E1"), (e2m, E2_R2[1:], 6, 6 / 34, "E2"))):
    for row, (metric, ylab) in enumerate((("recall_at_k", f"recall@{k}"), ("sign", "доля верных знаков"))):
        ax = axes[row, col]
        g = d.groupby(["method", "r2"])[metric].mean()
        methods = recall_e2_methods if tag == "E2" and metric == "recall_at_k" else TEST
        for meth in methods:
            key = meth.split("_")[0]
            if metric == "sign" and key == "pi":
                continue  # у PI нет знака
            y = g.loc[meth].reindex(levels)
            ax.plot(range(len(levels)), y.values, color=C[key], marker=MARK[key], markersize=6, label=NAME[key])
        if metric == "recall_at_k":
            ax.axhline(rnd, color=C["null"], ls=":", lw=1.2)
            ax.text(len(levels) - 1, rnd + 0.02, "случайный уровень", ha="right", va="bottom", fontsize=7.5, color=C["ink2"])
        else:
            ax.axhline(0.5, color=C["null"], ls=":", lw=1.2)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(ylab)
        if row == 0:
            ax.set_title(f"{tag}: {'синтетика, 4 истинных из 10' if tag == 'E1' else 'реальные признаки, 6 истинных из 34'}", loc="left")
        if row == 1:
            r2_axis(ax, levels)
axes[0, 0].legend(loc="lower right", ncol=2)
fig.suptitle("Восстановление признаков сигнала и направлений эффектов", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout()
save(fig, "fig02_recall_sign", "Recall и доля верных знаков против R² (E1, E2)")

# %% [markdown]
# ### Важность флага доступности аудио

# %%
f2 = read("e2_features")
f2 = f2[f2["r2"].isin([0.02, 0.08, 0.2]) & f2["method"].isin(["pi_test", "shap_test", "lime_test"])]
W2 = f2.pivot_table(index=["r2", "model", "seed", "method"], columns="feature", values="importance")
ratio = (W2["vk_can_see_audio"] / W2["vk_counter_audios"]).clip(lower=0).rename("ratio").reset_index()
med = ratio.groupby(["model", "method"])["ratio"].median()
fig, ax = plt.subplots(figsize=(6, 3.2))
models = ["hgb", "rf", "ridge"]
for j, meth in enumerate(["pi_test", "shap_test", "lime_test"]):
    key = meth.split("_")[0]
    xs = np.arange(3) + (j - 1) * 0.26
    vals = [med.loc[(m, meth)] for m in models]
    ax.bar(xs, vals, width=0.24, color=C[key], label=NAME[key])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=C["ink2"])
ax.set_xticks(range(3), [NAME[m] for m in models])
ax.set_ylabel("отношение важностей")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
ax.set_title("Важность флага доступности аудио относительно счётчика аудио", loc="left")
save(fig, "fig04_twin", "Двойник can_see_audio по моделям и методам (E2)")

# %% [markdown]
# ## 9. Что сохранилось
#
# - `fig11_e2_importance` — PI по трём моделям;
# - `fig12_e2_curves` — PDP/ALE HGB для отдельных эффектов;
# - `fig15_e2_interaction_curves` — PDP/ALE сигнала и HGB для взаимодействия;
# - `fig13_e2_shap_summary` — SHAP summary и heatmap;
# - `fig14_e2_local_outliers` — распределение отклонений, связь с истиной, признаки выбросов.
#
# Сравнения по агрегатам E1/E2: `fig01a_agreement_e12`, `fig02_recall_sign`, `fig04_twin`.
# Локальный кеш `results/e2_fig/` содержит в том числе матрицы признаков и SHAP.

# %%
print("\n".join(f"{name}: {title}" for name, title in SAVED))
