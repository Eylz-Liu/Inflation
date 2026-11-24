import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Courbe de taux zéro-coupon", layout="wide")

# ------------------------------------------------------------------
# 0. STYLE — noir, blanc, or ; typographie fine et très aérée
# ------------------------------------------------------------------
BLACK, WHITE, GOLD, GREY = "#000000", "#FFFFFF", "#B8975A", "#8A8A8A"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

.stApp { background: #FFFFFF; font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif; color: #000; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.5rem; max-width: 1400px; }

/* Titre */
.ysl-title { font-family: 'Jost', sans-serif; font-weight: 300; font-size: 2.1rem;
             letter-spacing: 0.38em; text-transform: uppercase; line-height: 1.25; margin: 0; }
.ysl-sub { font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 1.25rem;
           color: #8A8A8A; margin: 0.4rem 0 1.4rem 0; }
.ysl-rule { border: 0; border-top: 1px solid #000; margin: 0 0 2.2rem 0; }

/* Panneau de paramètres */
.ysl-panel { margin-bottom: 2.4rem; }
.ysl-panel h4 { font-weight: 400; font-size: 0.78rem; letter-spacing: 0.3em; text-transform: uppercase;
                margin: 0 0 0.9rem 0; padding-bottom: 0.7rem; border-bottom: 1px solid #000; }
.ysl-row { display: flex; justify-content: space-between; align-items: baseline;
           padding: 0.5rem 0; border-bottom: 1px solid #E6E6E6; font-size: 0.9rem; }
.ysl-row span:first-child { color: #8A8A8A; font-weight: 300; }
.ysl-row span:last-child  { font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 500; }
.ysl-gap { color: #B8975A; }

/* Barre latérale noire */
[data-testid="stSidebar"] { background: #000; }
[data-testid="stSidebar"] * { color: #F5F5F5; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-weight: 400; font-size: 0.78rem; letter-spacing: 0.3em; text-transform: uppercase;
    padding-bottom: 0.6rem; border-bottom: 1px solid #B8975A; margin-top: 1.6rem; }
[data-testid="stSidebar"] label p { font-weight: 300; letter-spacing: 0.04em; }
[data-testid="stSidebar"] input { background: #0E0E0E !important; color: #FFF !important; }
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0E0E0E !important; border: 1px solid #3A3A3A !important; border-radius: 0 !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] button { background: #0E0E0E !important; border-radius: 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #8A8A8A; font-style: italic; }

/* Bouton de téléchargement */
.stDownloadButton button { background: #000; color: #FFF; border: 1px solid #000; border-radius: 0;
    padding: 0.85rem 2rem; font-weight: 400; letter-spacing: 0.25em; text-transform: uppercase; font-size: 0.75rem; }
.stDownloadButton button:hover { background: #FFF; color: #000; border-color: #000; }
.stDownloadButton button:focus-visible { outline: 2px solid #B8975A; outline-offset: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 1. MODÈLE NELSON-SIEGEL
# ------------------------------------------------------------------
def nelson_siegel_rates(t, b0, b1, b2, alpha):
    """Taux zéro-coupon NS pour une maturité t (en années)."""
    t = np.asarray(t, dtype=float)
    x = t / alpha
    f1 = np.where(x == 0, 1.0, (1.0 - np.exp(-x)) / np.where(x == 0, 1.0, x))
    f2 = f1 - np.exp(-x)
    return b0 + b1 * f1 + b2 * f2


# ------------------------------------------------------------------
# 2. CALIBRATION
# ------------------------------------------------------------------
PARAM_NAMES = ["b0", "b1", "b2", "alpha"]
GUESS = {"b0": 0.03, "b1": -0.01, "b2": 0.0, "alpha": 2.0}
BOUNDS = {"b0": (-0.1, 0.2), "b1": (-0.2, 0.2), "b2": (-0.3, 0.3), "alpha": (0.5, 15.0)}
T_TARGETS = np.array([0.25, 10.0])


def calibrate_ns(t_targets, r_targets, fixed=None):
    """Calibre les paramètres non fixés pour coller aux points cibles."""
    fixed = fixed or {}
    free = [p for p in PARAM_NAMES if p not in fixed]
    if not free:
        return dict(fixed)

    def objective(x):
        params = {**fixed, **dict(zip(free, x))}
        return np.sum((nelson_siegel_rates(t_targets, **params) - r_targets) ** 2)

    res = minimize(
        objective,
        x0=[GUESS[p] for p in free],
        bounds=[BOUNDS[p] for p in free],
        method="L-BFGS-B",
    )
    return {**fixed, **dict(zip(free, res.x))}


def targets_from(inputs):
    """Cibles (3M, 10Y swap) à partir des inputs de marché."""
    return np.array([inputs["euribor_3m"], inputs["oat_10y"] - inputs["spread"]])


# ------------------------------------------------------------------
# 3. SITUATIONS PRÉDÉFINIES
# ------------------------------------------------------------------
SITUATIONS = {
    "31/12/2026": {"euribor_3m": 0.0257, "oat_10y": 0.0411, "spread": 0.0066},
    "31/12/2027": {"euribor_3m": 0.0220, "oat_10y": 0.0380, "spread": 0.0050},
    "Personnalisé": {"euribor_3m": 0.0257, "oat_10y": 0.0411, "spread": 0.0066},
}
COLORS = {"31/12/2026": BLACK, "31/12/2027": GOLD, "Personnalisé": "#5A5A5A"}

# ------------------------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------------------------
st.markdown(
    '<p class="ysl-title">Courbe de taux<br>zéro-coupon</p>'
    '<p class="ysl-sub">Modèle Nelson-Siegel</p><hr class="ysl-rule">',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Situation")
    situation = st.selectbox("Date d'évaluation", list(SITUATIONS.keys()))
    s = SITUATIONS[situation]

    st.header("Marché")
    euribor_3m = st.number_input("Taux court 3M", value=s["euribor_3m"], format="%.4f", key=f"e_{situation}")
    oat_10y = st.number_input("OAT 10 ans", value=s["oat_10y"], format="%.4f", key=f"o_{situation}")
    spread = st.number_input("Spread France", value=s["spread"], format="%.4f", key=f"s_{situation}")
    max_months = int(st.number_input("Maturité max (mois)", value=600, min_value=12, max_value=1200, step=12))

    inputs = {"euribor_3m": euribor_3m, "oat_10y": oat_10y, "spread": spread}

    st.header("Paramètres")
    mode = st.radio("Mode", ["Calibration automatique", "Saisie manuelle"])

    fixed, compare = {}, False
    if mode == "Calibration automatique":
        st.caption("2 cibles pour 4 paramètres : fixer alpha et b2 pour une solution stable.")
        if st.checkbox("Fixer alpha", value=True):
            fixed["alpha"] = st.number_input("alpha", 0.5, 15.0, 2.0, step=0.1)
        if st.checkbox("Fixer b2 (courbure)", value=True):
            fixed["b2"] = st.number_input("b2", -0.30, 0.30, 0.0, step=0.001, format="%.4f")
        compare = st.checkbox("Comparer 2026 et 2027", value=False)
        params = calibrate_ns(T_TARGETS, targets_from(inputs), fixed)
    else:
        params = {
            "b0": st.slider("Beta 0 (niveau LT)", -0.10, 0.20, 0.035, step=0.0001, format="%.4f"),
            "b1": st.slider("Beta 1 (pente)", -0.20, 0.20, -0.012, step=0.0001, format="%.4f"),
            "b2": st.slider("Beta 2 (courbure)", -0.30, 0.30, 0.002, step=0.0001, format="%.4f"),
            "alpha": st.slider("Alpha", 0.5, 15.0, 2.0, step=0.1),
        }

# ------------------------------------------------------------------
# 5. COURBES À AFFICHER
# ------------------------------------------------------------------
months = np.arange(1, max_months + 1)
years = months / 12.0

if compare:
    to_plot = {
        label: (SITUATIONS[label], calibrate_ns(T_TARGETS, targets_from(SITUATIONS[label]), fixed))
        for label in ["31/12/2026", "31/12/2027"]
    }
else:
    to_plot = {situation: (inputs, params)}

curves = {label: nelson_siegel_rates(years, **p) for label, (_, p) in to_plot.items()}

# ------------------------------------------------------------------
# 6. GRAPHIQUE & PARAMÈTRES
# ------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    for label, (inp, p) in to_plot.items():
        c = COLORS.get(label, BLACK)
        tg = targets_from(inp)
        ax.plot(months, curves[label] * 100, color=c, linewidth=1.6, label=f"Courbe {label}")
        ax.scatter([3], [tg[0] * 100], facecolors=WHITE, edgecolors=c, s=70, linewidths=1.4, zorder=5)
        ax.scatter([120], [tg[1] * 100], facecolors=WHITE, edgecolors=c, marker="D", s=60, linewidths=1.4, zorder=5)
    # légende des cibles, une seule fois
    ax.scatter([], [], facecolors=WHITE, edgecolors=BLACK, s=70, label="Cible 3M")
    ax.scatter([], [], facecolors=WHITE, edgecolors=BLACK, marker="D", s=60, label="Cible swap 10 ans")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BLACK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=BLACK, labelsize=9, length=3)
    ax.set_xlabel("Maturité (mois)", fontsize=9, color=BLACK, labelpad=10)
    ax.set_ylabel("Taux zéro-coupon (%)", fontsize=9, color=BLACK, labelpad=10)
    ax.set_xlim(0, max_months)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    for label, (inp, p) in to_plot.items():
        tg = targets_from(inp)
        gap3 = (nelson_siegel_rates(0.25, **p) - tg[0]) * 1e4
        gap10 = (nelson_siegel_rates(10.0, **p) - tg[1]) * 1e4
        rows = "".join(f'<div class="ysl-row"><span>{k}</span><span>{p[k]:.6f}</span></div>' for k in PARAM_NAMES)
        rows += f'<div class="ysl-row"><span>Écart 3M</span><span class="ysl-gap">{gap3:.2f} bps</span></div>'
        rows += f'<div class="ysl-row"><span>Écart 10 ans</span><span class="ysl-gap">{gap10:.2f} bps</span></div>'
        st.markdown(f'<div class="ysl-panel"><h4>{label}</h4>{rows}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 7. EXPORT EXCEL (sans afficher le tableau)
# ------------------------------------------------------------------
df = pd.DataFrame({"Mois": months, "Maturité (années)": years})
for label, c in curves.items():
    df[f"Taux ZC {label} (%)"] = (c * 100).round(4)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="CTZ")

suffix = "comparaison" if compare else situation.replace("/", "")
st.download_button(
    "Télécharger la courbe",
    data=buffer.getvalue(),
    file_name=f"Courbe_CTZ_{suffix}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)