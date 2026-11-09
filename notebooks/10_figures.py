# %% [markdown]
# # 10. Главные графики для текста
#
# Только из кеша `results/*.csv`, модели не обучаются и объяснения не пересчитываются.
# Файлы: `results/figures/figNN_*.png` и `.pdf`. Ссылки на графики — в `docs/summary.md`.
#
# **Персональные данные:** ноутбук не читает `data/`. Все величины — агрегаты по признакам,
# прогонам и перестановкам.
#
# Запуск: `uv run python notebooks/10_figures.py`.

# %%
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from xai_consistency.data import CONTINUOUS  # noqa: E402  (только списки имён признаков)

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read(name):
    return pd.read_csv(RESULTS / f"{name}.csv", keep_default_na=False, na_values=[""], low_memory=False)


# Цвета методов те же, что в ноутбуках 06–08 (палитра проверена на различимость при дальтонизме).
C = {"pi": "#2a78d6", "shap": "#eb6834", "lime": "#1baf7a", "pdp": "#eda100", "ale": "#e87ba4",
     "lime_cat": "#6250d6", "rf": "#2a78d6", "hgb": "#eb6834", "ridge": "#1baf7a",
     "extra": "#2a78d6", "emo": "#eb6834", "null": "#9a9893", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
MARK = {"pi": "o", "shap": "s", "lime": "^", "pdp": "D", "ale": "v", "rf": "o", "hgb": "s", "ridge": "^"}
NAME = {"pi": "PI", "shap": "SHAP", "lime": "LIME", "pdp": "PDP", "ale": "ALE",
        "rf": "RF", "hgb": "HGB", "ridge": "Ridge"}
SHORT = {"vk_counter_friends": "друзья", "vk_post_avg_likes": "лайки", "has_university": "has_university",
         "vk_counter_audios": "аудио", "vk_can_see_audio": "can_see_audio"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
SAVED = []


def save(fig, name, title):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    SAVED.append((name, title))
    print(f"{name}: {title}")


def r2_axis(ax, levels):
    """Уровни R² на равномерной оси (они не равноотстоят)."""
    ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
    ax.set_xlabel("R² сигнала")


TEST = ["pi_test", "shap_test", "lime_test", "pdp_test", "ale_test"]
GROUP = ["shap_test", "lime_test", "pdp_test", "ale_test"]


def split_pairs(p):
    """Средний ρ внутри SHAP/LIME/PDP/ALE и PI с остальными по каждому прогону."""
    p = p[p["a"].isin(TEST) & p["b"].isin(TEST)]
    is_pi = (p["a"] == "pi_test") | (p["b"] == "pi_test")
    return p.assign(group=np.where(is_pi, "pi", "within"))

# %% [markdown]
# ## 1. Согласие методов против силы сигнала (E1, E2, E3)

# %%
e1p = split_pairs(read("e1_pairs"))
e2p = split_pairs(read("e2_pairs").query("set == 'continuous20'"))
E1_R2, E2_R2 = [0.0, 0.02, 0.08, 0.2, 0.5], [0.0, 0.02, 0.08, 0.2]

# E3: ρ между методами по 20 непрерывным признакам в каждом прогоне
e3 = read("e3_importance")
e3 = e3[e3["method"].isin(TEST) & e3["feature"].isin(CONTINUOUS)]
W3 = e3.pivot_table(index=["trait", "model", "kind", "idx", "feature"], columns="method", values="importance")


def e3_pairs(g):
    rho = {(a, b): stats.spearmanr(g[a], g[b])[0] for a, b in combinations(TEST, 2)}
    return pd.Series({"pi": np.mean([v for k, v in rho.items() if "pi_test" in k]),
                      "within": np.mean([v for k, v in rho.items() if "pi_test" not in k])})


e3pair = W3.groupby(level=["trait", "model", "kind", "idx"]).apply(e3_pairs).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, gridspec_kw={"width_ratios": [5, 4, 3.2]})
for ax, p, levels, title in ((axes[0], e1p, E1_R2, "E1: синтетика"), (axes[1], e2p, E2_R2, "E2: полусинтетика")):
    m = p[p["model"].isin(["rf", "hgb"])].groupby(["group", "model", "r2"])["spearman"].mean()
    for grp, col, lab in (("within", C["shap"], "внутри SHAP, LIME, PDP, ALE"), ("pi", C["pi"], "PI с остальными")):
        for mod, ls in (("rf", "-"), ("hgb", "--")):
            y = m.loc[(grp, mod)].reindex(levels)
            ax.plot(range(len(levels)), y.values, ls=ls, color=col, marker=MARK[mod],
                    label=f"{lab}, {NAME[mod]}")
    r2_axis(ax, levels)
    ax.set_title(title, loc="left")
axes[0].set_ylabel("средний ρ Спирмена между методами")
axes[0].axhline(0, color=C["ink2"], lw=0.8)
axes[1].axhline(0, color=C["ink2"], lw=0.8)
ax = axes[2]
cats = [("null", "нуль"), ("emo", "emo, CV"), ("extra", "extra, CV")]
for j, (key, lab) in enumerate(cats):
    sel = e3pair[e3pair["kind"] == "null"] if key == "null" else e3pair[(e3pair["trait"] == key) & (e3pair["kind"] == "cv")]
    for mod, dx in (("rf", -0.12), ("hgb", 0.12)):
        v = sel[sel["model"] == mod][["within", "pi"]].mean()
        ax.plot(j + dx, v["within"], marker=MARK[mod], color=C["shap"], ls="none")
        ax.plot(j + dx, v["pi"], marker=MARK[mod], color=C["pi"], ls="none")
ax.set_xticks(range(3), [c[1] for c in cats])
ax.set_xlim(-0.5, 2.5)
ax.axhline(0, color=C["ink2"], lw=0.8)
ax.set_title("E3: реальные черты", loc="left")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Согласие SHAP, LIME, PDP и ALE не зависит от сигнала, согласие с PI растёт вместе с ним",
             x=0.01, y=1.03, ha="left", fontsize=10.5)
fig.text(0.01, -0.25, "Деревья (RF, HGB); у Ridge четыре метода совпадают по построению (|β|·SD). "
         "E1 — среднее по 5 сценариям, E2 и E3 — 20 непрерывных признаков. E3: нуль — 100 перестановок цели.",
         fontsize=7.5, color=C["ink2"])
save(fig, "fig01_agreement", "Согласие методов против R² (E1, E2) и на реальных чертах (E3)")

# %% [markdown]
# ## 2. Recall и знаки против R² (E1, E2)

# %%
e1m = read("e1_methods")
e1m = e1m[e1m["method"].isin(TEST) & e1m["scenario"].isin(["independent", "correlated ρ=0.5"])]
e2m = read("e2_methods")
e2m = e2m[e2m["method"].isin(TEST)]
fig, axes = plt.subplots(2, 2, figsize=(8.5, 6), sharex="col")
for col, (d, levels, k, rnd, tag) in enumerate(((e1m, E1_R2[1:], 4, 0.4, "E1"), (e2m, E2_R2[1:], 6, 6 / 34, "E2"))):
    for row, (metric, ylab) in enumerate((("recall_at_k", f"recall@{k}"), ("sign", "доля верных знаков"))):
        ax = axes[row, col]
        g = d.groupby(["method", "r2"])[metric].mean()
        for meth in TEST:
            key = meth.split("_")[0]
            if metric == "sign" and key == "pi":
                continue  # у PI нет знака
            y = g.loc[meth].reindex(levels)
            ax.plot(range(len(levels)), y.values, color=C[key], marker=MARK[key], label=NAME[key])
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
fig.suptitle("Знаки находятся раньше, чем полный набор признаков", x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.text(0.01, -0.03, "Среднее по моделям и повторам. E1 — сценарии independent и correlated ρ = 0.5. "
         "Знак — доля монотонных истинных признаков с верным знаком; у PI знака нет. При R² = 0 истинных признаков нет.", fontsize=7.5, color=C["ink2"])
fig.tight_layout()
save(fig, "fig02_recall_sign", "Recall и доля верных знаков против R² (E1, E2)")

# %% [markdown]
# ## 3. PDP против ALE у истины и у модели (E2, признаки взаимодействия)

# %%
cur = read("e2_curves")
cur = cur[cur["feature"].isin(["vk_counter_photos", "vk_wall_posts_total"]) & (cur["r2"] == 0.2)]
rows = [("pdp_true", "ale_true", "истина: PDP и ALE"),
        ("ale_true", "apley_ale", "истина: ALE и теория ½ρc(z² − 1)"),
        ("pdp", "ale", "модель: PDP и ALE"),
        ("pdp", "pdp_true", "модель против истины: PDP"),
        ("ale", "ale_true", "модель против истины: ALE")]
fig, ax = plt.subplots(figsize=(7.5, 3.3))
for i, (a, b, lab) in enumerate(rows):
    for mod, dy in (("rf", -0.17), ("hgb", 0.17)):
        v = cur[(cur["a"] == a) & (cur["b"] == b) & (cur["model"] == mod)]["ccc"].mean()
        ax.barh(i + dy, v, height=0.3, color=C[mod], label=NAME[mod] if i == 0 else None)
        ax.text(v + 0.01, i + dy, f"{v:.2f}", va="center", fontsize=7.5, color=C["ink2"])
ax.set_yticks(range(len(rows)), [r[2] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("CCC кривых (1 — совпадают)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", title="модель (у истины одинаково)", title_fontsize=7.5)
ax.set_title("У истины PDP и ALE расходятся, у почти аддитивных моделей совпадают", loc="left")
fig.text(0.01, -0.06, "E2, R² 0.2, photos и wall_posts (произведение в f, r = 0.54), среднее по 30 повторам.",
         fontsize=7.5, color=C["ink2"])
save(fig, "fig03_pdp_ale", "PDP против ALE для истины и модели (E2)")

# %% [markdown]
# ## 4. Двойник по моделям (E2)

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
ax.set_ylabel("важность двойника / важность audios")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
ax.set_title("Важность двойника — свойство модели, методы её честно показывают", loc="left")
fig.text(0.01, -0.08, "E2: can_see_audio (r = 0.88 с audios) в истинную функцию не входит, истинное отношение 0. "
         "Медиана по 90 прогонам (R² 0.02–0.2).", fontsize=7.5, color=C["ink2"])
save(fig, "fig04_twin", "Двойник can_see_audio по моделям и методам (E2)")

# %% [markdown]
# ## 5. `extra` против `emo` против нуля (E3)

# %%
runs3 = read("e3_runs")
t1000 = read("e3_trees_1000")
obs3 = e3[e3["kind"] == "obs"]
full3 = read("e3_importance")


def z_scores(trait, model, method):
    """z наблюдаемой важности относительно нуля того же признака (extra деревья: 1000 перестановок)."""
    o = full3[(full3["trait"] == trait) & (full3["model"] == model) & (full3["kind"] == "obs") & (full3["method"] == method)]
    o = o.set_index("feature")["importance"]
    if trait == "extra" and model in ("rf", "hgb"):
        n = t1000[(t1000["model"] == model) & (t1000["method"] == method)]
    else:
        n = full3[(full3["trait"] == trait) & (full3["model"] == model) & (full3["kind"] == "null") & (full3["method"] == method)]
    n = n.pivot(index="idx", columns="feature", values="importance")[o.index]
    return (o - n.mean()) / n.std()


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
rng = np.random.default_rng(0)
for i, mod in enumerate(["ridge", "rf", "hgb"]):
    for j, (lab, sel, col) in enumerate((("нуль", (runs3["model"] == mod) & (runs3["kind"] == "null") & (runs3["trait"] == "extra"), C["null"]),
                                         ("emo", (runs3["model"] == mod) & (runs3["kind"] == "cv") & (runs3["trait"] == "emo"), C["emo"]),
                                         ("extra", (runs3["model"] == mod) & (runs3["kind"] == "cv") & (runs3["trait"] == "extra"), C["extra"]))):
        v = runs3.loc[sel, "r2_test"].to_numpy()
        x = i + (j - 1) * 0.27 + rng.uniform(-0.07, 0.07, len(v))
        ax.scatter(x, v, s=9, color=col, alpha=0.55, lw=0, label=lab if i == 0 else None)
        ax.plot([i + (j - 1) * 0.27 - 0.1, i + (j - 1) * 0.27 + 0.1], [np.mean(v)] * 2, color=C["ink"], lw=1.6)
ax.axhline(0, color=C["ink2"], lw=0.8)
ax.set_xticks(range(3), ["Ridge", "RF", "HGB"])
ax.set_ylabel("R² на тесте")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", ncol=3, markerscale=1.8)
ax.set_title("а) Качество: 25 фолдов CV и 100 перестановок", loc="left")

ax = axes[1]
pos = 0
labels = []
for mod in ("rf", "hgb"):
    for meth in ("pi_test", "shap_test"):
        for trait in ("extra", "emo"):
            z = z_scores(trait, mod, meth)
            y = pos + rng.uniform(-0.18, 0.18, len(z))
            ax.scatter(z.values, y, s=14, color=C[trait], alpha=0.75, lw=0)
            for f, dy in (("vk_counter_friends", 7), ("vk_post_avg_likes", -7)):
                if z[f] > 2.5:
                    ax.annotate(SHORT[f], (z[f], y[z.index.get_loc(f)]), xytext=(3, dy), textcoords="offset points",
                                fontsize=7, va="center", color=C["ink2"])
            labels.append(f"{trait}: {NAME[mod]}, {NAME[meth.split('_')[0]]}")
            pos += 1
ax.axvline(0, color=C["ink2"], lw=0.8)
ax.set_yticks(range(pos), labels)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("z важности относительно нуля этого признака")
ax.set_title("б) 34 признака: важность против перестановок", loc="left")
fig.suptitle("extra отличается от нуля двумя признаками, emo — ничем", x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.text(0.01, -0.06, "Нуль: перемешанная цель, то же разбиение 80/20. z — только для наглядности; "
         "значимость в тексте по эмпирическому p на 1000 перестановках (extra) и BH. emo — 100 перестановок.",
         fontsize=7.5, color=C["ink2"])
fig.tight_layout()
save(fig, "fig05_e3_null", "extra против emo против нуля: качество и важности (E3)")

# %% [markdown]
# ## 6. Устойчивость рангов между фолдами не доказывает сигнал (E3)

# %%
W = full3[full3["method"].isin(TEST)].pivot_table(index=["trait", "model", "kind", "idx", "method"],
                                                  columns="feature", values="importance")


def mean_pair_rho(block):
    block = block.dropna(axis=1)  # у PDP и ALE нет флагов
    r = stats.spearmanr(block.to_numpy().T)[0]
    return r[np.triu_indices(len(block), 1)].mean()


st = []
for (t, m, k, meth), g in W.groupby(level=["trait", "model", "kind", "method"]):
    if k == "obs" or (k == "null" and t != "extra"):
        continue
    g = g.iloc[:25]  # нуль: 25 перестановок, как фолдов
    st.append({"trait": t, "model": m, "kind": k, "method": meth.split("_")[0], "rho": mean_pair_rho(g)})
st = pd.DataFrame(st)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, mod in zip(axes, ("rf", "hgb")):
    for j, (lab, sel) in enumerate((("нуль", (st["kind"] == "null")), ("emo, CV", (st["trait"] == "emo") & (st["kind"] == "cv")),
                                     ("extra, CV", (st["trait"] == "extra") & (st["kind"] == "cv")))):
        for i, meth in enumerate(["pi", "shap", "lime", "pdp", "ale"]):
            v = st[sel & (st["model"] == mod) & (st["method"] == meth)]["rho"].iloc[0]
            ax.bar(j + (i - 2) * 0.16, v, width=0.15, color=C[meth], label=NAME[meth] if j == 0 else None)
    ax.set_xticks(range(3), ["нуль", "emo, CV", "extra, CV"])
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_title(NAME[mod], loc="left")
axes[0].set_ylabel("средний ρ важностей между прогонами")
axes[1].legend(loc="upper left", ncol=5, fontsize=7.5, bbox_to_anchor=(-1.1, -0.12))
fig.suptitle("Ранги SHAP и LIME устойчивы и без сигнала, устойчивость PI следует за сигналом", x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.text(0.01, -0.2, "E3. CV — 25 фолдов повторной 5×5 CV; нуль — 25 перестановок цели extra на одном разбиении. "
         "34 признака у PI, SHAP, LIME; 20 непрерывных у PDP и ALE.", fontsize=7.5, color=C["ink2"])
save(fig, "fig06_stability", "Устойчивость рангов: extra, emo и нуль (E3)")

# %% [markdown]
# ## 7. LIME на флагах с `categorical_features` и без (E4)

# %%
e4 = read("e4_e2_importance")
e4 = e4[e4["model"].isin(["rf", "hgb"]) & e4["variant"].isin(["pi", "shap", "lime", "lime_cat"])]
R4 = e4.pivot_table(index=["r2", "model", "seed", "variant"], columns="feature", values="importance").rank(axis=1, ascending=False)
top = (R4["has_university"] <= 6).groupby(level=["r2", "model", "variant"]).mean()
VAR = [("pi", "PI", C["pi"]), ("shap", "SHAP", C["shap"]), ("lime", "LIME, гаусс", C["lime"]),
       ("lime_cat", "LIME categorical", C["lime_cat"])]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, mod in zip(axes, ("hgb", "rf")):
    for i, (v, lab, col) in enumerate(VAR):
        vals = [top.loc[(r, mod, v)] for r in (0.0, 0.08, 0.2)]
        xs = np.arange(3) + (i - 1.5) * 0.2
        ax.bar(xs, vals, width=0.18, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", fontsize=6.5, color=C["ink2"])
    ax.set_xticks(range(3), ["0", "0.08", "0.2"])
    ax.set_xlabel("R² сигнала")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title(NAME[mod], loc="left")
axes[0].set_ylabel("доля повторов с has_university в top-6")
axes[0].legend(loc="upper left", fontsize=7.5)
fig.suptitle("Гауссово возмущение LIME размывает ступеньку флага; categorical возвращает его к уровню SHAP",
             x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.text(0.01, -0.08, "E2/E4, 30 повторов. LIME categorical: эффект перехода 0 → 1 пересчитан в наклон на 1 SD. "
         "has_university — 9% дисперсии истинной функции.", fontsize=7.5, color=C["ink2"])
save(fig, "fig07_lime_flags", "LIME на флаге has_university: гауссово возмущение и categorical (E4)")

# %% [markdown]
# ## 8. Порог доли дисперсии на признак (E4)

# %%
pr = read("e4_probe")
P = pr.pivot_table(index=["probe", "s", "model", "seed", "variant"], columns="feature", values="importance")
PR = P.rank(axis=1, ascending=False)
PROBES = [("vk_counter_friends", "друзья (непрерывный)", "-"), ("vk_counter_audios", "аудио (есть двойник)", "--"),
          ("has_university", "has_university (флаг)", ":")]
S = [0.0, 0.005, 0.01, 0.02, 0.04]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, v in zip(axes, ("pi", "shap")):
    for f, lab, ls in PROBES:
        y = []
        for s in S:
            src = PR.xs("none", level="probe") if s == 0 else PR.xs((f, s), level=["probe", "s"])
            y.append((src.xs(v, level="variant")[f] <= 6).mean())
        ax.plot(range(len(S)), y, ls=ls, color=C[v], marker="o", label=lab)
    ax.set_xticks(range(len(S)), ["0", "0.5%", "1%", "2%", "4%"])
    ax.set_xlabel("доля дисперсии y у признака-пробы")
    ax.set_ylim(0, 1.03)
    ax.set_title(NAME[v], loc="left")
axes[0].set_ylabel("доля повторов с пробой в top-6")
axes[0].legend(loc="lower right", fontsize=7.5)
fig.suptitle("Одиночный признак находится уже при 0.5–1% дисперсии y, флаг — позже", x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.text(0.01, -0.08, "E4, пункт 5: один сигнальный признак среди 34 реальных, 800 объектов обучения, "
         "среднее по Ridge, RF, HGB и 30 повторам.", fontsize=7.5, color=C["ink2"])
save(fig, "fig08_threshold", "Порог обнаружения одиночного признака (E4)")

# %% [markdown]
# ## 9. Local R² LIME: синтетика и реальные черты (E4)

# %%
path = RESULTS / "e4_local_r2_real.csv"
if not path.exists():
    print("Нет results/e4_local_r2_real.csv: сначала ноутбук 09, раздел 6. График 9 пропущен.")
else:
    e1r, e2r, lr = read("e4_e1_runs"), read("e4_e2_runs"), read("e4_local_r2_real")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for d, levels, tag, ls in ((e1r, E1_R2[:1] + E1_R2[2:], "E1", "-"), (e2r, [0.0, 0.08, 0.2], "E2", "--")):
        for mod in ("rf", "hgb"):
            y = d[d["model"] == mod].groupby("r2")["local_r2_lime"].mean().reindex(levels)
            xs = [E1_R2.index(v) for v in levels]
            ax.plot(xs, y.values, ls=ls, color=C[mod], marker=MARK[mod], label=f"{tag}, {NAME[mod]}")
    r2_axis(ax, E1_R2)
    ax.set_ylabel("медиана local R² по объектам")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("а) E1 и E2: растёт с сигналом", loc="left")
    ax = axes[1]
    for i, mod in enumerate(("rf", "hgb")):
        for j, (lab, sel, col) in enumerate((("нуль", (lr["kind"] == "null") & (lr["trait"] == "extra"), C["null"]),
                                             ("emo", (lr["kind"] == "cv") & (lr["trait"] == "emo"), C["emo"]),
                                             ("extra", (lr["kind"] == "cv") & (lr["trait"] == "extra"), C["extra"]))):
            v = lr.loc[sel & (lr["model"] == mod), "local_r2_median"].to_numpy()
            x0 = i + (j - 1) * 0.27
            ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=9, color=col, alpha=0.55, lw=0,
                       label=lab if i == 0 else None)
            ax.plot([x0 - 0.1, x0 + 0.1], [np.mean(v)] * 2, color=C["ink"], lw=1.6)
    ax.set_xticks(range(2), ["RF", "HGB"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, markerscale=1.8, fontsize=7.5)
    ax.set_title("б) E3: фолды CV и перестановки цели", loc="left")
    fig.suptitle("Local R² LIME — одна из немногих локальных метрик, которая следует за сигналом",
                 x=0.01, y=1.04, ha="left", fontsize=10.5)
    fig.text(0.01, -0.14, "Медиана local R² по 200 тестовым объектам. Сравнивать только с собственным нулём: "
             "без сигнала он 0.3–0.5, а не 0. Нуль в (б) — 100 перестановок extra на разбиении 0.", fontsize=7.5, color=C["ink2"])
    fig.tight_layout()
    save(fig, "fig09_local_r2", "Local R² LIME против сигнала: E1, E2 и реальные черты")

# %%
print("\nСохранено:")
for n, t in SAVED:
    print(f"  results/figures/{n}.png|pdf — {t}")
