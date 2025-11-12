import io, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Peer Trial Analyzer", layout="wide")

VALID_CATEGORIES = set()

# VALID_FEEDBACK = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}

# ---------- Reading files ----------
def read_any(uploaded_file, sheet_name=None):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        if sheet_name is None:
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("Only support CSV or XLSX")

# ---------- Cleaning ----------
def clean_and_prepare(df_raw):
    df = df_raw.copy()
    need_cols = ["Trial", "Prompt", "SelectedPeer", "FeedbackType"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Extract Category (all uppercase words; single letters are allowed)
    df["Category"] = df["Prompt"].str.findall(r'\b[A-Z]+\b').str.join('_')
    df["Category"] = df["Category"].astype(str).str.title()
    Category = set(df["Category"])
    df.loc[~df["Category"].isin(Category), "Category"] = pd.NA

    df["Trial"] = pd.to_numeric(df["Trial"], errors="coerce")
    df["SelectedPeer"] = pd.to_numeric(df["SelectedPeer"], errors="coerce").astype("Int64")
    df = df.fillna(0)
    # df["FeedbackType"] = (
    #     df["FeedbackType"].astype(str).str.strip().str.lower().map(VALID_FEEDBACK).fillna("Neutral")
    # )

    # df = df.dropna(subset=["Trial", "SelectedPeer", "Category"])

    # df = df[df["SelectedPeer"].isin([1, 2, 3, 4])]
    # df["Trial"] = df["Trial"].astype(int)
    # df = df.sort_values("Trial").reset_index(drop=True)
    return df, Category

# ---------- Statistical Matrix ----------
def build_matrices(df):
    emotions = ["negative", "neutral", "positive"]
    ct = pd.crosstab(
        index=df["Trial"],
        columns=[df["Category"], df["SelectedPeer"], df["FeedbackType"]],
    ).sort_index()
    ct_cum = ct.cumsum()

    df_matrix = ct_cum.copy()
    df_matrix.columns = [f"{c}_{int(p)}_{f}" for c, p, f in df_matrix.columns]

    wanted_cols = []
    for cate in VALID_CATEGORIES:
        for peer in [1, 2]:
            for emo in emotions[1:]:
                wanted_cols.append(f"{cate}_{peer}_{emo}")
        for peer in [3, 4]:
            for emo in emotions[:2]:
                wanted_cols.append(f"{cate}_{peer}_{emo}")

    for col in wanted_cols:
        if col not in df_matrix.columns:
            df_matrix[col] = 0
    df_matrix = df_matrix.reindex(columns=wanted_cols)
    return df_matrix

# ---------- Percentage ----------
def to_percent(df_matrix, window=None):
    base = df_matrix.copy()
    if window and window > 1:
        base = base.rolling(window, min_periods=max(1, window//4)).sum()

    pct = df_matrix.copy()
    for cat in VALID_CATEGORIES:
        for peer in [1,2,3,4]:
            cols = [c for c in pct.columns if c.startswith(f"{cat}_{peer}_")]
            if not cols:
                continue
            denom = pct[cols].sum(axis=1).replace(0, pd.NA)
            pct[cols] = pct[cols].div(denom, axis=0).fillna(0)
    pct.index.name = "Trial"
    return pct

def real_ratio(df):
    peer_ratio_counts = df.groupby(["SelectedPeer", "FeedbackType"]).size().reset_index(name="Count")
    peer_ratio_counts["Ratio"] = peer_ratio_counts.groupby("SelectedPeer")["Count"].transform(lambda x: x / x.sum())
    first_ratios = peer_ratio_counts.groupby('SelectedPeer').first()['Ratio']
    return dict(zip(first_ratios.index.astype(int), first_ratios.astype(float)))

def sanity_checks(df, df_matrix, pct, categories, ft_order=('Negative','Neutral','Positive')):
    print("=== unique FeedbackType (raw) ===")
    print(df['FeedbackType'].head(10))
    print(df['FeedbackType'].astype(str).str.strip().str.title().value_counts(dropna=False))

    print("\n=== categories (from data) ===")
    print(categories)

    print("\n=== df_matrix columns sample ===")
    print(df_matrix.columns[:10].tolist(), "...")

    print("\n=== any nonzero in df_matrix? ===")
    print(df_matrix.sum().sum())

    print("\n=== pct(top 10) ===")
    print((pct.head(-10)))

    print("\n=== pct nonzero by column (top 10) ===")
    print((pct.sum(axis=0).sort_values(ascending=False).head(10)))



# ---------- Plot ----------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_grid(pct, df_matrix, ratio_lines=None):
    """
    画 4x4 (category x peer) 堆叠面积图。
    pct:        百分比 DataFrame（列名形如 Category_Peer_FeedbackType）
    df_matrix:  计数累计 DataFrame（用于计算每个子图的 total）
    ratio_lines:可选，{peer: ratio}，为每个 peer 画一条横线
    """
    ft_order = ['negative', 'neutral', 'positive']
    colors = {'negative': "#1f77b4", 'neutral': "#2ca02c", 'positive': "#f08930ab"}
    peers = [1, 2, 3, 4]

    nrows, ncols = (len(VALID_CATEGORIES), len(peers))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=True, sharey=True)
    # 统一成二维数组，避免 nrows/ncols 为 1 时下标问题
    axes = np.atleast_2d(axes)

    trials = pct.index

    if len(trials) == 0:
        # 没有数据也返回一个空图，防止崩
        plt.tight_layout()
        return fig

    # 计算 total（来自累计计数的最后一行）
    total_map = {}
    for cate in VALID_CATEGORIES:
        for peer in peers:
            cols = [f"{cate}_{peer}_{ft}" for ft in ft_order]
            exist_cols = [c for c in cols if c in df_matrix.columns]
            total = int(df_matrix[exist_cols].iloc[-1].sum()) if (len(df_matrix) and exist_cols) else 0
            total_map[(cate, peer)] = total

    for c, cate in enumerate(VALID_CATEGORIES):
        for p, peer in enumerate(peers):
            ax = axes[c, p]
            cols = [f"{cate}_{peer}_{ft}" for ft in ft_order]
            sub = pct.reindex(columns=cols, fill_value=0)

            # 堆叠面积图
            ax.stackplot(
                trials,
                *[sub[name].values for name in cols],
                labels=ft_order,
                colors=[colors[ft] for ft in ft_order],
                alpha=0.85
            )

            # 标题 & 轴标签
            if c == 0:
                ax.set_title(f"Peer {peer}")
            if p == 0:
                ax.set_ylabel(cate, rotation=0, labelpad=30, va='center')

            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            if c == nrows - 1:
                ax.set_xlabel("Trial", fontsize=12, labelpad=20)

            ax.set_xlim(trials.min(), trials.max())
            ax.tick_params(axis='x', labelbottom=True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

            ax.text(
                0.5, -0.22, f"Total: {total_map[(cate, peer)]}",
                transform=ax.transAxes, ha='center', va='top', fontsize=12, clip_on=False)

            # 参考线：优先用 ratio_lines；否则按原逻辑
            
            if isinstance(ratio_lines, dict) and peer in ratio_lines:
                ax.axhline(ratio_lines[peer], color='white', linestyle='--', linewidth=1, alpha=0.9)
            else:
                lo, hi = (0.25, 0.75)
                if peer in (2, 3):
                    ax.axhline(hi, color='white', linestyle='--', linewidth=1, alpha=0.8)
                if peer in (1, 4):
                    ax.axhline(lo, color='white', linestyle='--', linewidth=1, alpha=0.8)

    # 图例
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[ft]) for ft in ft_order]
    fig.legend(
        handles, ft_order, title="FeedbackType",
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06)
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


# ---------- UI ----------
st.title("Peer Trial Analyzer")
st.write("Upload trial_log (CSV/XLSX) and generate 4×4 category×peer stacked area chart. Percentage = Count / Total")

uploaded = st.file_uploader("Please upload the trial log file (CSV or XLSX)", type=["csv","xlsx","xls"])
sheet = None
df = None

if uploaded is not None and uploaded.name.lower().endswith(("xlsx","xls")):
    try:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select Sheet (if excel)", xls.sheet_names, index=0)
    except Exception:
        sheet = None

# win = st.slider("滚动窗口长度（trial 数）", min_value=1, max_value=60, value=20, step=1)
# win = st.slider("Rolling window length (number of trials)", min_value=1, max_value=60, value=20, step=1)
use_realratio = st.checkbox("Use real ratio (from data) as reference line", value=True)
# pick_mode = st.selectbox("real ratio 选择方式",
#                          ["first(按Neg/Neu/Pos顺序取第一)", "positive(取Positive)", "max(取最高占比)"],
#                          index=0)

if uploaded:
    try:
        df_raw = read_any(uploaded, sheet)
        df, VALID_CATEGORIES = clean_and_prepare(df_raw)
        st.success(f"Data rows: {len(df)}")
        with st.expander("Check out the first 10 rows after cleaning"):
            st.dataframe(df.head(10))

        df_matrix = build_matrices(df)
        # pct = to_percent(df_matrix, window=win if win>1 else None)
        pct = to_percent(df_matrix, window=None)


        # —— 生成 ratio_lines 字典 ——
        if use_realratio:
            ratio_lines = real_ratio(df)
        else:
            ratio_lines = None

        sanity_checks(df, df_matrix, pct, VALID_CATEGORIES)
        fig = plot_grid(pct, df_matrix, ratio_lines=ratio_lines)
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
        buf.seek(0)
        st.download_button("Download", buf, file_name="peer_grid.png", mime="image/png")

    except Exception as e:
        st.error(f"Error: {e}")
