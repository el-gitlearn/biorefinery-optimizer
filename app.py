import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import linprog

st.set_page_config(page_title="Biorefinery Pathway Optimizer", page_icon="🌾", layout="wide",
                   initial_sidebar_state="auto")

# =====================================================================
# DESIGN TOKENS
# =====================================================================
INK, MUTED, CANVAS, LINE = "#14281D", "#5B6B60", "#F5F8F2", "#DCE5D6"
LEAF, STRAW, TEAL, LOSS = "#1F6F4A", "#D9A21B", "#0E7C86", "#B5523B"
PATH_COLOR = {
    "Pelletization": "#C98A12",
    "Combustion for power": "#CF4B2C",
    "Anaerobic digestion (CBG)": "#2F8F6B",
    "Slow pyrolysis (biochar)": "#46433E",
    "Cellulosic ethanol (2G)": "#3A6EA5",
}
SHORT = {
    "Pelletization": "Pellets",
    "Combustion for power": "Power",
    "Anaerobic digestion (CBG)": "Biogas (CBG)",
    "Slow pyrolysis (biochar)": "Biochar",
    "Cellulosic ethanol (2G)": "Ethanol",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;700&display=swap');
:root { --ink:#14281D; --muted:#5B6B60; --canvas:#F5F8F2; --line:#DCE5D6; --leaf:#1F6F4A; --straw:#D9A21B; --teal:#0E7C86; }
html, body, .stApp, p, label, li, input, textarea, button, [data-baseweb="select"] div, .stMarkdown, .stCaption {
  font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, [data-testid="stMetricValue"], .hero-name, .brand, .fig b, .page-title {
  font-family: 'Bricolage Grotesque', 'DM Sans', system-ui, sans-serif !important;
  letter-spacing: -0.01em;
}
.stApp { background: var(--canvas); color: var(--ink); }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* sidebar */
[data-testid="stSidebar"] { background: #EBF1E6; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { font-size: 1.05rem; font-weight: 700; color: var(--leaf); margin: 0 0 .2rem 0; }
.brand-sub { color: var(--muted); font-size: .82rem; margin: 0 0 1.1rem 0; line-height: 1.35; }
.sb-h { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: .98rem; color: var(--ink);
        margin: 1.1rem 0 .25rem 0; padding-bottom: .25rem; border-bottom: 1px solid var(--line); }
[data-testid="stSidebar"] label p { font-size: .86rem; font-weight: 500; color: var(--ink); }

/* page header */
.page-title { font-size: clamp(1.7rem, 3.2vw, 2.5rem); font-weight: 700; line-height: 1.12; margin: .2rem 0 .5rem 0; color: var(--ink); max-width: 26ch; }
.page-sub { color: var(--muted); font-size: 1.02rem; line-height: 1.5; max-width: 62ch; margin: 0 0 1.4rem 0; }

/* hero result */
.hero { display: flex; flex-wrap: wrap; gap: 1.6rem 2.4rem; align-items: center; justify-content: space-between;
        background: var(--ink); color: #F3F7EF; border-radius: 18px; padding: 1.7rem 2rem; margin-bottom: 1.6rem; }
.hero-k { color: #B9CBB9; font-size: .92rem; margin-bottom: .25rem; }
.hero-name { font-size: clamp(1.8rem, 3.4vw, 2.7rem); font-weight: 700; line-height: 1.05; }
.hero-why { color: #CFE0CF; margin-top: .55rem; font-size: .98rem; max-width: 44ch; line-height: 1.45; }
.hero-figs { display: flex; flex-wrap: wrap; gap: 1.2rem 2.2rem; }
.fig b { display: block; font-size: 1.65rem; font-weight: 700; color: #FFFFFF; line-height: 1.15; }
.fig span { color: #B9CBB9; font-size: .85rem; }
.fig.gold b { color: #F0C45A; } .fig.teal b { color: #6FD0D8; }

/* ranked list */
.rank { margin: .2rem 0 1rem 0; }
.rk-row { display: grid; grid-template-columns: 1.6rem 1fr; gap: .2rem .7rem; padding: .8rem 0; border-bottom: 1px solid var(--line); }
.rk-row:last-child { border-bottom: none; }
.rk-n { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: var(--muted); font-size: 1.1rem; }
.rk-row.best .rk-n { color: var(--leaf); }
.rk-name { font-weight: 700; display: flex; align-items: center; flex-wrap: wrap; gap: .3rem .5rem; }
.dot { width: .72rem; height: .72rem; border-radius: 50%; display: inline-block; flex: none; }
.rk-goal { color: var(--muted); font-size: .82rem; font-weight: 400; margin-left: .2rem; }
.rk-bar { height: 6px; background: #E2EAD9; border-radius: 4px; margin: .45rem 0 .35rem 0; overflow: hidden; }
.rk-bar i { display: block; height: 100%; border-radius: 4px; }
.rk-fig { color: var(--muted); font-size: .86rem; }
.rk-fig b { color: var(--ink); font-weight: 500; }

/* section titles inside tabs */
.sec { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.25rem; margin: 1.8rem 0 .15rem 0; color: var(--ink); }
.sec-sub { color: var(--muted); font-size: .95rem; margin: 0 0 .7rem 0; max-width: 70ch; }

/* tabs */
button[data-baseweb="tab"] { font-size: 1rem; font-weight: 500; padding: .7rem 1.1rem; }
button[data-baseweb="tab"][aria-selected="true"] { font-weight: 700; color: var(--ink); }
[data-baseweb="tab-highlight"] { background-color: var(--leaf) !important; height: 3px; }
[data-baseweb="tab-border"] { background-color: var(--line); }

/* metrics, expanders, buttons */
[data-testid="stMetric"] { background: transparent; padding: .2rem 0; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .88rem; }
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; color: var(--ink); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #FFFFFF; }
[data-testid="stExpander"] summary p { font-weight: 500; }
.stDownloadButton button { border-radius: 10px; border: 1px solid var(--leaf); color: var(--leaf); background: #FFFFFF; font-weight: 500; }
.stDownloadButton button:hover { background: var(--leaf); color: #FFFFFF; border-color: var(--leaf); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
div[data-baseweb="radio"] label { font-weight: 500; }
.note { color: var(--muted); font-size: .88rem; line-height: 1.5; }
@media (max-width: 640px) { .hero { padding: 1.2rem; } .fig b { font-size: 1.35rem; } }
/* ---- premium polish ---- */
html { scroll-behavior: smooth; }
.hero, .rk-row, .stat, .fig b, [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.hero { animation: rise .6s cubic-bezier(.2,.7,.2,1) both; box-shadow: 0 24px 48px -28px rgba(20,40,29,.6); }
.hero-k::before { content: ""; display: inline-block; width: 22px; height: 3px; border-radius: 2px; background: var(--straw);
                  margin-right: .6rem; vertical-align: middle; }
.card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px; padding: .3rem 1.3rem;
        box-shadow: 0 1px 2px rgba(20,40,29,.04), 0 14px 30px -20px rgba(20,40,29,.25); }
.rk-row { transition: background .2s ease; border-radius: 10px; padding-left: .5rem; padding-right: .5rem; animation: rise .5s ease both; }
.rk-row:nth-child(2) { animation-delay: .05s; } .rk-row:nth-child(3) { animation-delay: .1s; }
.rk-row:nth-child(4) { animation-delay: .15s; } .rk-row:nth-child(5) { animation-delay: .2s; }
.rk-row:hover { background: #F4F8F0; }
.pill { white-space: nowrap; display: inline-block; font-size: .72rem; font-weight: 500; color: var(--muted); border: 1px solid var(--line);
        border-radius: 999px; padding: .04rem .55rem; margin-left: .4rem; background: #FFFFFF; }
.pill.best { color: #FFFFFF; background: var(--leaf); border-color: var(--leaf); }
.rk-bar { height: 7px; }
[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px; padding: .6rem .6rem .2rem .6rem;
        box-shadow: 0 1px 2px rgba(20,40,29,.04), 0 14px 30px -20px rgba(20,40,29,.25); }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: .6rem 0 .8rem 0; }
.stat { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; animation: rise .5s ease both;
        box-shadow: 0 1px 2px rgba(20,40,29,.04); }
.stat b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 700; line-height: 1.15; color: var(--ink); }
.stat span { color: var(--muted); font-size: .85rem; }
.stat.gold b { color: #9A6F00; } .stat.teal b { color: #0A6A73; }
.brand-row { display: flex; align-items: center; gap: .6rem; margin-bottom: .3rem; }
.brand-row .brand { margin: 0; }
.foot { color: var(--muted); font-size: .82rem; margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid var(--line); }
/* route picker as pills */
div[role="radiogroup"] { gap: .5rem; flex-wrap: wrap; }
div[role="radiogroup"] label { background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px; padding: .3rem 1rem; cursor: pointer;
                               transition: border-color .15s ease, background .15s ease; }
div[role="radiogroup"] label:hover { border-color: var(--leaf); }
[data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stRadioOption"][data-selected="true"] { background: var(--ink) !important; border-color: var(--ink) !important; }
[data-testid="stRadioOption"][data-selected="true"] p { color: #FFFFFF !important; }
/* inputs */
[data-testid="stSidebar"] [data-baseweb="input"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within > div, [data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div {
  border-color: var(--leaf); }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: #FFFFFF; }
@media (max-width: 640px) { .stats { grid-template-columns: 1fr; } }

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height=360, margin=None):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, system-ui, sans-serif", size=13, color=INK),
        margin=margin or dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font_size=13, font_family="DM Sans, system-ui, sans-serif"),
        title_font=dict(family="Bricolage Grotesque, DM Sans, sans-serif", size=17, color=INK),
    )
    try:
        fig.update_layout(barcornerradius=6)
    except Exception:
        pass
    fig.update_xaxes(gridcolor="#E7EDE2", zerolinecolor="#9DB09F", linecolor=LINE)
    fig.update_yaxes(gridcolor="#E7EDE2", zerolinecolor="#9DB09F", linecolor=LINE)
    return fig


def money(v):
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:,.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:,.1f} lakh"
    return f"{sign}₹{a:,.0f}"


def neg(v):
    return "₹0" if abs(v) < 0.5 else f"-₹{abs(v):,.0f}"


def out_amount(unit, amount):
    if unit == "kWh":
        return f"{amount / 1000:,.0f} MWh"
    if unit == "kg":
        return f"{amount / 1000:,.0f} t"
    if unit == "L":
        return f"{amount / 1000:,.0f} kL"
    return f"{amount:,.0f} t"




# =====================================================================
# DATA LEGEND   [C] cited from a source   [D] derived from cited values
#               [A] assumption (placeholder, replace with plant data when available)
# All quantities are per tonne of DRY residue.
# =====================================================================
KCAL_PER_MJ = 238.85

FEEDSTOCKS = {
    "Rice straw": dict(
        hhv=14.6,          # [C] MJ/kg, range 14.08-15.09 (Springer rice straw chapter)
        c_frac=0.38,       # [A] carbon fraction of dry straw
        biochar_yield=0.38,   # [D] interpolated from 33.9-49.9 wt% (SBJChem 2018)
        ret_c=0.45,        # [C] 45 % of straw carbon retained in biochar (Bioresour. Technol. 155, 2014)
        etoh_l=143.0,      # [C/D] IOCL Panipat: 2.1 lakh t dry straw -> 3 crore L (ThePrint)
        cbg_kg=105.0,      # [D] Sangrur plant: 33 t/day CBG from 1.1 lakh t straw/yr; 8-9 t straw per t CBG
        power_capex_mw=7.44e7,   # [C] ₹per MW, rice-straw air-cooled (CERC proposal 2026)
    ),
    "Sugarcane bagasse": dict(
        hhv=17.9,          # [C] MJ/kg dry (17.88 +/- 0.48)
        c_frac=0.47,       # [C] ultimate analysis 46.8-48.0 % C
        biochar_yield=0.25,   # [A] literature gives 16 % in a fast free-fall reactor
        ret_c=0.35,        # [A]
        etoh_l=174.0,      # [D] rice-straw yield scaled by carbohydrate fraction (0.51 vs 0.42 L/kg theoretical)
        cbg_kg=None,       # no reliable data yet
        power_capex_mw=6.38e7,   # [C] ₹per MW, biomass Rankine (lower end, CERC proposal 2026)
    ),
}

# ---- constants -------------------------------------------------------
GRID_EF = 0.727          # [C] kg CO2/kWh, CEA CO2 Baseline Database v20, FY2023-24 (RE-adjusted average)
COAL_EF = 94.6           # [C] kg CO2/GJ, IPCC 2006 default (bituminous coal)
CH4_CO2 = 2.75 * 0.95    # [D] kg CO2 per kg CBG (stoichiometry, CBG >= 95 % methane)
ETOH_AVOIDED = 0.66 * 2.31   # [D] kg CO2 per L ethanol (0.66 L petrol-equivalent x 2.31 kg CO2/L)
PELLET_BENCH = 2.27      # [C] ₹per 1000 kcal, MoP benchmark price, northern region, ex-GST
SHR = 4000.0             # [C] kcal/kWh station heat rate, regulator norm
AUX = 0.10               # [C] auxiliary consumption, regulator norm
OM_POWER_MW = 5.47e6     # [C] ₹per MW per year, CERC 2024 O&M norm (FY2024-25)
PLF = 0.80               # [C] plant load factor norm
ETOH_PRICE = 66.61       # [C] ₹/L weighted average ex-mill price, ESY 2025-26 (range 57.97-71.86)
CBG_PRICE = 54.0         # [C] ₹/kg SATAT price slab (minimum guaranteed 46 ₹/kg to 2029)
POWER_TARIFF = 5.22      # [C] ₹/kWh, PSERC approved biomass plant tariff FY2025-26 (CERC generic proposals: 9.6-11.6)


DISCOUNT_RATE, PROJECT_LIFE = 0.10, 20   # [A] used for the plant capital charge


def crf(rate, years):
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def build_pathways(feed):
    f = FEEDSTOCKS[feed]
    rows = []
    hhv_kcal = f["hhv"] * KCAL_PER_MJ          # kcal per kg

    # 1. Pelletization -------------------------------------------------
    y = 0.90                                      # [D] 1.05-1.15 t feed per t pellet (PelletRates)
    rows.append(dict(
        Pathway="Pelletization", Goal="Heat / power", Product="Pellets", Unit="t", Yield=y,
        Price=PELLET_BENCH * hhv_kcal,            # [C] benchmark ₹/1000 kcal x calorific value
        Capex=1e7 / (6000 / y),                   # [C] ₹1 crore per tonne/hour plant (DownToEarth); 6000 h/yr [A]
        Opex=1200.0,                              # [A] power 90-150 kWh/t pellet + labour + maintenance
        Avoided=y * f["hhv"] * COAL_EF,           # [D] cross-check: NTPC reports 1,283 kg CO2 per t pellet
        ProcEmis=y * 102,                         # [C] NTPC: 63-102 kg CO2 per t pellet (upper value)
        Basis="Price = MoP benchmark ₹2.27 per 1000 kcal x calorific value. Capex ₹1 crore per t/h. "
              "Avoided CO2 = pellet energy x coal factor (NTPC reports 1,283 kg/t). Opex is an assumption."))

    # 2. Combustion for power -------------------------------------------
    gross = hhv_kcal * 1000 / SHR                 # kWh gross per tonne
    net = gross * (1 - AUX)
    fuel_per_mw = 8760 * PLF * 1000 / gross       # tonnes of feed per MW per year
    rows.append(dict(
        Pathway="Combustion for power", Goal="Heat / power", Product="Electricity", Unit="kWh", Yield=net,
        Price=POWER_TARIFF,
        Capex=f["power_capex_mw"] / fuel_per_mw,  # [C] CERC capital cost norm
        Opex=OM_POWER_MW / fuel_per_mw,           # [C] CERC O&M norm
        Avoided=net * GRID_EF,                    # [C] CEA grid factor
        ProcEmis=10.0,                            # [A] handling
        Basis="Net kWh = calorific value / heat rate 4000 kcal/kWh x (1 - 10 % auxiliary). "
              "Capex and O&M from CERC norms. Tariff = Punjab regulator FY2025-26. Avoided CO2 = CEA grid factor."))

    # 3. Anaerobic digestion -> CBG (rice straw only) ---------------------
    if f["cbg_kg"]:
        cbg_t_per_day_feed = (1000 / f["cbg_kg"]) * 330
        capex = 6.5e7 / cbg_t_per_day_feed        # [C] ₹6-7 crore per tonne/day CBG, paddy straw (CSE)
        rows.append(dict(
            Pathway="Anaerobic digestion (CBG)", Goal="Gaseous fuel", Product="CBG", Unit="kg", Yield=f["cbg_kg"],
            Price=CBG_PRICE, Capex=capex, Opex=0.05 * capex,     # O&M 5 % of capex [A]
            Coprod=1.75 * 1000 * 3.0 * 0.5,       # manure: 1.75 t/t straw [C], ₹3/kg [C], 50 % sold [A]
            Avoided=f["cbg_kg"] * CH4_CO2,
            ProcEmis=40.0,                        # [A]
            Basis="Yield from the Sangrur paddy-straw plant (about 100-110 kg CBG per t). Price = SATAT slab. "
                  "Capex ₹6.5 crore per t/day CBG. Manure (FOM) income: 1.75 t per t straw x ₹3/kg x 50 % sold (assumption)."))

    # 4. Slow pyrolysis -> biochar ------------------------------------------
    stored = f["c_frac"] * f["ret_c"] * 44 / 12 * 1000 * 0.80   # kg CO2e per t feed; 80 % permanence [A]
    rows.append(dict(
        Pathway="Slow pyrolysis (biochar)", Goal="Soil / carbon storage", Product="Biochar", Unit="t",
        Yield=f["biochar_yield"], Price=12000.0, Capex=0.0, Opex=2500.0,
        Avoided=stored, ProcEmis=60.0,
        Basis="Yield from literature. Price: no reliable Indian market price (assumption). Cost about ₹2,500 per t "
              "residue (Vietnam study, low confidence). Stored carbon = C fraction x retained share x 80 % permanence."))

    # 5. Cellulosic (2G) ethanol -----------------------------------------------
    capex = 9e9 / 210000                          # [C] IOCL Panipat: over ₹900 crore, 2.1 lakh t/yr
    rows.append(dict(
        Pathway="Cellulosic ethanol (2G)", Goal="Liquid fuel", Product="Ethanol", Unit="L", Yield=f["etoh_l"],
        Price=ETOH_PRICE, Capex=capex, Opex=0.05 * capex,        # O&M 5 % of capex [A]; enzymes not included
        Avoided=f["etoh_l"] * ETOH_AVOIDED, ProcEmis=150.0,      # [A]
        Basis="Yield and capex from the IOCL Panipat 2G plant (over ₹900 crore, 2.1 lakh t straw -> 3 crore L). "
              "Price = weighted average ex-mill ethanol price ESY 2025-26. Enzyme and chemical costs not included."))
    out = pd.DataFrame(rows)
    if "Coprod" not in out:
        out["Coprod"] = 0.0
    out["Coprod"] = out["Coprod"].fillna(0.0)
    return out





# =====================================================================
# SIDEBAR  (inputs)
# =====================================================================
with st.sidebar:
    st.markdown('<div class="brand-row"><svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">'
                '<rect width="32" height="32" rx="9" fill="#14281D"/>'
                '<path d="M9 22c0-7 5-12 14-12 0 8-5 13-12 13-1 0-2-.3-2-1z" fill="#D9A21B"/>'
                '<path d="M9.5 22.5c3-4 6-7 10-9.5" stroke="#14281D" stroke-width="1.6" stroke-linecap="round"/></svg>'
                '<div class="brand">Biorefinery Pathway Optimizer</div></div>'
                '<div class="brand-sub">Choose a residue and your goals. Results update as you change anything.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sb-h">Your residue</div>', unsafe_allow_html=True)
    feed = st.selectbox("Crop residue", list(FEEDSTOCKS.keys()))
    qty = st.number_input("Quantity (tonnes dry residue per year)", min_value=1.0, value=1000.0, step=100.0)
    feed_cost = st.number_input("Delivered residue cost (₹ per tonne)", min_value=0.0, value=3000.0, step=100.0,
                                help="What you pay to get the residue to your plant. About ₹3,000/t in one biogas company example (Verbio, via Mongabay). "
                                     "IOCL pays ₹1,800/t at the depot (ThePrint).")
    st.markdown('<div class="sb-h">Your goals</div>', unsafe_allow_html=True)
    goals_all = ["Heat / power", "Gaseous fuel", "Liquid fuel", "Soil / carbon storage"]
    goals = st.multiselect("What do you want to produce?", goals_all, default=goals_all)
    w = st.slider("Priority: CO₂ first ... profit first", 0, 100, 50,
                  help="0 = only CO₂ avoided matters, 100 = only profit matters")
    st.markdown('<div class="sb-h">Advanced</div>', unsafe_allow_html=True)
    with st.expander("Expenses, transport and tax"):
        extra = st.number_input("Your other expenses (₹ per tonne of residue)", min_value=0.0, value=0.0, step=100.0,
                                help="Anything else you pay, such as labour, storage or handling. Subtracted from profit.")
        transport = st.number_input("Transport charges (₹ per tonne of residue)", min_value=0.0, value=0.0, step=100.0,
                                    help="Enter your own transport cost, if any. Subtracted from profit.")
        tax = st.number_input("Tax (% of profit, if any)", min_value=0.0, max_value=60.0, value=0.0, step=1.0,
                              help="Applied only when profit is positive. Leave at 0 if there is no tax.")

base = build_pathways(feed)

with st.sidebar:
    with st.expander("Edit prices and costs"):
        pr, cx, ox = [], [], []
        for _, r in base.iterrows():
            st.markdown(f"**{r['Pathway']}**")
            pr.append(st.number_input(f"Price (₹ per {r['Unit']})", value=float(round(r["Price"], 2)), key=f"{feed}{r['Pathway']}p"))
            cx.append(st.number_input("Capex (₹ per annual t residue)", value=float(round(r["Capex"], 0)), key=f"{feed}{r['Pathway']}c"))
            ox.append(st.number_input("Operating cost (₹ per t residue)", value=float(round(r["Opex"], 0)), key=f"{feed}{r['Pathway']}o"))
        base["Price"], base["Capex"], base["Opex"] = pr, cx, ox

# =====================================================================
# CALCULATIONS
# =====================================================================
df = base.copy()
df["Revenue (Rs/t)"] = df["Yield"] * df["Price"] + df["Coprod"]
df["Capital charge (Rs/t)"] = df["Capex"] * crf(DISCOUNT_RATE, PROJECT_LIFE)
df["Transport (Rs/t)"] = transport
df["Other expenses (Rs/t)"] = extra
pre_tax = (df["Revenue (Rs/t)"] - df["Capital charge (Rs/t)"] - df["Opex"] - feed_cost
           - df["Transport (Rs/t)"] - df["Other expenses (Rs/t)"])
df["Tax (Rs/t)"] = pre_tax.clip(lower=0) * tax / 100
df["Profit (Rs/t)"] = pre_tax - df["Tax (Rs/t)"]
df["Net CO2 avoided (kg/t)"] = df["Avoided"] - df["ProcEmis"]
df["Total profit (Rs lakh/yr)"] = df["Profit (Rs/t)"] * qty / 1e5
df["Total CO2 avoided (t/yr)"] = df["Net CO2 avoided (kg/t)"] * qty / 1000
df = df[df["Goal"].isin(goals)].reset_index(drop=True)


def norm(s):
    return pd.Series([1.0] * len(s)) if s.max() == s.min() else (s - s.min()) / (s.max() - s.min())


# =====================================================================
# HEADER
# =====================================================================
st.markdown('<h1 class="page-title">Which route turns crop residue into the most value?</h1>'
            '<p class="page-sub">Compare conversion routes on profit and CO<sub>2</sub>, using Indian plant data and prices. '
            'Change the inputs on the left and everything updates.</p>', unsafe_allow_html=True)

if df.empty:
    st.warning("Pick at least one goal in the sidebar to see results.")
    st.stop()

df["Score"] = (w / 100) * norm(df["Profit (Rs/t)"]) + (1 - w / 100) * norm(df["Net CO2 avoided (kg/t)"])
df = df.sort_values("Score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", range(1, len(df) + 1))
best = df.iloc[0]

# ---- hero -----------------------------------------------------------------
top_profit = df["Profit (Rs/t)"].idxmax()
top_co2 = df["Net CO2 avoided (kg/t)"].idxmax()
if len(df) == 1:
    why = "The only route that matches your goals."
elif top_profit == 0 and top_co2 == 0:
    why = "Leads on both profit and CO₂ avoided with your inputs."
elif top_profit == 0:
    why = f"Highest profit. {df.loc[top_co2, 'Pathway']} avoids more CO₂."
elif top_co2 == 0:
    why = f"Highest CO₂ benefit. {df.loc[top_profit, 'Pathway']} earns more."
else:
    why = "Best balance of profit and CO₂ at your priority."
prod_label = best["Product"] if best["Product"] == "CBG" else best["Product"].lower()
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-k">Best route for your goals</div>
    <div class="hero-name">{best['Pathway']}</div>
    <div class="hero-why">{why}</div>
  </div>
  <div class="hero-figs">
    <div class="fig gold"><b>{money(best['Total profit (Rs lakh/yr)'] * 1e5)}</b><span>profit per year</span></div>
    <div class="fig teal"><b>{best['Total CO2 avoided (t/yr)']:,.0f} t</b><span>CO<sub>2</sub> avoided per year</span></div>
    <div class="fig"><b>{out_amount(best['Unit'], best['Yield'] * qty)}</b><span>{prod_label} per year</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Compare routes", "Optimize the mix"])

# =====================================================================
# TAB 1  Compare routes
# =====================================================================
with tab1:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown('<div class="sec" style="margin-top:.2rem">Ranked for your priority</div>'
                    '<div class="sec-sub">Longer bar means a better overall fit.</div>', unsafe_allow_html=True)
        rows_html = ""
        for _, r in df.iterrows():
            col = PATH_COLOR[r["Pathway"]]
            wbar = max(4, round(r["Score"] * 100))
            rows_html += f"""
<div class="rk-row {'best' if r['Rank'] == 1 else ''}">
  <div class="rk-n">{int(r['Rank'])}</div>
  <div>
    <div class="rk-name"><span class="dot" style="background:{col}"></span>{r['Pathway']}{'<span class="pill best">Best fit</span>' if r['Rank'] == 1 else ''}<span class="pill">{r['Goal']}</span></div>
    <div class="rk-bar"><i style="width:{wbar}%;background:{col}"></i></div>
    <div class="rk-fig"><b>{money(r['Profit (Rs/t)'])}</b> profit per t &nbsp;·&nbsp; <b>{r['Net CO2 avoided (kg/t)']:,.0f} kg</b> CO<sub>2</sub> avoided per t</div>
  </div>
</div>"""
        st.markdown(f'<div class="card rank">{rows_html}</div>', unsafe_allow_html=True)
        if len(df) > 1:
            run = df.iloc[1]
            note = f"Runner-up: {run['Pathway']}."
            st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec" style="margin-top:.2rem">Profit against CO<sub>2</sub>, route by route</div>'
                    '<div class="sec-sub">Gold bars show profit, teal bars show CO<sub>2</sub> avoided, both per tonne of residue.</div>',
                    unsafe_allow_html=True)
        d = df.iloc[::-1]
        names = [SHORT[p] for p in d["Pathway"]]
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.22,
                            column_titles=["CO<sub>2</sub> avoided (kg per t)", "Profit (₹ per t)"])
        fig.add_trace(go.Bar(y=names, x=d["Net CO2 avoided (kg/t)"], orientation="h", marker_color=TEAL,
                             text=[f"{v:,.0f}" for v in d["Net CO2 avoided (kg/t)"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}<br>%{x:,.0f} kg CO₂ per t<extra></extra>", showlegend=False), 1, 1)
        fig.add_trace(go.Bar(y=names, x=d["Profit (Rs/t)"], orientation="h",
                             marker_color=[STRAW if v >= 0 else LOSS for v in d["Profit (Rs/t)"]],
                             text=[f"{v:,.0f}" for v in d["Profit (Rs/t)"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}<br>₹%{x:,.0f} per t<extra></extra>", showlegend=False), 1, 2)
        co2_max = max(d["Net CO2 avoided (kg/t)"].max(), 1)
        pf_min, pf_max = min(d["Profit (Rs/t)"].min(), 0), max(d["Profit (Rs/t)"].max(), 1)
        fig.update_xaxes(range=[co2_max * 1.3, 0], row=1, col=1)
        fig.update_xaxes(range=[pf_min * 2.3 if pf_min < 0 else 0, pf_max * 1.3], row=1, col=2)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(side="right", row=1, col=1, tickfont=dict(size=13))   # route names sit in the gap between the panels
        fig.update_layout(bargap=0.38)
        style_fig(fig, height=max(300, 70 * len(d) + 90), margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="sec">Where the money goes</div>'
                '<div class="sec-sub">Pick a route to see how its sales turn into profit, per tonne of residue.</div>',
                unsafe_allow_html=True)
    pick = st.radio("Route", list(df["Pathway"]), horizontal=True, label_visibility="collapsed",
                    format_func=lambda p: SHORT[p])
    r = df[df["Pathway"] == pick].iloc[0]
    wf = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * 7 + ["total"],
        x=["Product sales", "Plant capital", "Operating cost", "Residue cost", "Transport", "Your expenses", "Tax", "Profit"],
        y=[r["Revenue (Rs/t)"], -r["Capital charge (Rs/t)"], -r["Opex"], -feed_cost, -r["Transport (Rs/t)"],
           -r["Other expenses (Rs/t)"], -r["Tax (Rs/t)"], 0],
        text=[f"₹{r['Revenue (Rs/t)']:,.0f}", neg(r['Capital charge (Rs/t)']), neg(r['Opex']),
              neg(feed_cost), neg(r['Transport (Rs/t)']), neg(r['Other expenses (Rs/t)']),
              neg(r['Tax (Rs/t)']), f"₹{r['Profit (Rs/t)']:,.0f}"],
        textposition="outside", cliponaxis=False,
        increasing=dict(marker=dict(color=LEAF)), decreasing=dict(marker=dict(color=LOSS)),
        totals=dict(marker=dict(color=INK)), connector=dict(line=dict(color="#9DB09F", width=1)),
        hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>"))
    wf.update_layout(showlegend=False, yaxis_title="₹ per tonne of residue")
    style_fig(wf, height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(wf, width="stretch", config={"displayModeBar": False})
    st.markdown(f'<div class="note"><b>How it is calculated:</b> {r["Basis"]}</div>', unsafe_allow_html=True)

    with st.expander("See the full table"):
        show = df[["Rank", "Pathway", "Goal", "Product", "Yield", "Unit", "Profit (Rs/t)", "Net CO2 avoided (kg/t)", "Score"]].copy()
        show = show.rename(columns={"Profit (Rs/t)": "Profit (₹/t)", "Net CO2 avoided (kg/t)": "CO₂ avoided (kg/t)"})
        show["Yield"] = show["Yield"].round(2)
        show["Profit (₹/t)"] = show["Profit (₹/t)"].round(0)
        show["CO₂ avoided (kg/t)"] = show["CO₂ avoided (kg/t)"].round(0)
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f")})

# =====================================================================
# TAB 2  Optimize the mix
# =====================================================================
def solve_lp(d, qty, cap, primary, co2_min=None, second=None):
    """Allocate residue tonnes x_i to pathways (linear program, SciPy HiGHS).
    primary / second: 'profit' or 'co2' (second is a tie-break)."""
    prof = d["Profit (Rs/t)"].to_numpy(dtype=float)
    co2 = d["Net CO2 avoided (kg/t)"].to_numpy(dtype=float)
    n = len(d)
    coef = {"profit": prof, "co2": co2}
    bounds = [(0, cap * qty)] * n
    A, b = [np.ones(n)], [qty]                       # total residue <= available
    if co2_min is not None:
        A.append(-co2)                               # CO2 avoided >= target
        b.append(-co2_min)

    r = linprog(-coef[primary], A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not r.success:
        return None
    if second:
        best = -r.fun
        A2 = A + [-coef[primary]]                    # keep the primary objective at its optimum
        b2 = b + [-(best - 1e-6 * max(1.0, abs(best)))]
        r2 = linprog(-coef[second], A_ub=np.array(A2), b_ub=np.array(b2), bounds=bounds, method="highs")
        if r2.success:
            r = r2
    xs = np.maximum(r.x, 0.0)
    return dict(x=xs.tolist(), profit=float(prof @ xs), co2=float(co2 @ xs))



with tab2:
    st.markdown('<div class="sec" style="margin-top:.2rem">Split the residue across routes</div>'
                '<div class="sec-sub">A linear program decides how many tonnes go to each route. It maximizes profit while meeting a '
                'CO<sub>2</sub> target (epsilon-constraint method, Mavrotas, 2009) and traces the trade-off curve.</div>',
                unsafe_allow_html=True)
    cap_pct = st.slider("Most residue any one route can take (%)", 20, 100, 40, step=10,
                        help="No single route can absorb all residue in practice (plant capacity, market demand). "
                             "This is an assumption. Change it freely.")
    cap = cap_pct / 100
    if cap * len(df) < 1:
        st.warning("These routes cannot absorb all the residue at this limit. Some residue stays unused. "
                   "Raise the limit or pick more goals.")
    a = solve_lp(df, qty, cap, "profit", second="co2")
    b = solve_lp(df, qty, cap, "co2", second="profit")
    if a is None or b is None:
        st.error("The optimization could not be solved. Check your inputs.")
    else:
        co2_a, co2_b = a["co2"], b["co2"]
        if co2_b - co2_a < 1e-6:
            st.success("No trade-off here: the most profitable split is also the best for CO₂.")
            chosen = a
        else:
            pts = []
            for k in range(9):
                eps = co2_a + (co2_b - co2_a) * k / 8
                rr = solve_lp(df, qty, cap, "profit", co2_min=eps - 1e-6)
                if rr:
                    pts.append(rr)
            t = st.slider("Move along the trade-off curve", 0, 100, 50,
                          help="0 = maximum profit, 100 = maximum CO₂ avoided")
            eps_sel = co2_a + (co2_b - co2_a) * t / 100
            chosen = solve_lp(df, qty, cap, "profit", co2_min=eps_sel - 1e-6)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=[p["profit"] / 1e5 for p in pts], y=[p["co2"] / 1000 for p in pts],
                                      mode="lines+markers", name="Best possible splits", line=dict(color=LEAF, width=3),
                                      marker=dict(size=9, color=LEAF),
                                      hovertemplate="Profit ₹%{x:,.1f} lakh<br>%{y:,.0f} t CO₂<extra></extra>"))
            fig3.add_trace(go.Scatter(x=[chosen["profit"] / 1e5], y=[chosen["co2"] / 1000], mode="markers", name="Your choice",
                                      marker=dict(size=18, color=STRAW, line=dict(color=INK, width=2)),
                                      hovertemplate="Profit ₹%{x:,.1f} lakh<br>%{y:,.0f} t CO₂<extra></extra>"))
            fig3.update_layout(title="Trade-off curve", xaxis_title="Profit (₹ lakh per year)",
                               yaxis_title="CO₂ avoided (t per year)", legend=dict(orientation="h", y=-0.2))
            style_fig(fig3, height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

        st.markdown(f"""
<div class="stats">
  <div class="stat gold"><b>{money(chosen['profit'])}</b><span>profit per year</span></div>
  <div class="stat teal"><b>{chosen['co2'] / 1000:,.0f} t</b><span>CO<sub>2</sub> avoided per year</span></div>
  <div class="stat"><b>{sum(chosen['x']):,.0f} of {qty:,.0f} t</b><span>residue used</span></div>
</div>""", unsafe_allow_html=True)
        if sum(chosen["x"]) < qty - 1:
            st.markdown(f'<div class="note">{qty - sum(chosen["x"]):,.0f} t of residue is left unprocessed: routes that lose money '
                        'are skipped unless they are needed to reach the CO₂ target, or the limit on each route is reached.</div>',
                        unsafe_allow_html=True)

        alloc = pd.DataFrame({
            "Pathway": df["Pathway"],
            "Residue allocated (t/yr)": [round(v, 1) for v in chosen["x"]],
            "Share (%)": [round(100 * v / qty, 1) for v in chosen["x"]],
            "Profit (₹ lakh/yr)": [round(df.loc[i, "Profit (Rs/t)"] * chosen["x"][i] / 1e5, 2) for i in range(len(df))],
            "CO₂ avoided (t/yr)": [round(df.loc[i, "Net CO2 avoided (kg/t)"] * chosen["x"][i] / 1000, 1) for i in range(len(df))],
        })
        alloc = alloc[alloc["Residue allocated (t/yr)"] > 0.05].reset_index(drop=True)
        st.markdown('<div class="sec">Recommended split</div>', unsafe_allow_html=True)
        fb = go.Figure()
        for _, rr in alloc.iterrows():
            fb.add_trace(go.Bar(y=["Residue"], x=[rr["Residue allocated (t/yr)"]], orientation="h", name=SHORT[rr["Pathway"]],
                                marker_color=PATH_COLOR[rr["Pathway"]],
                                text=[f"{SHORT[rr['Pathway']]}<br>{rr['Share (%)']:.0f}%"], textposition="inside",
                                insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=13),
                                hovertemplate=f"{rr['Pathway']}<br>%{{x:,.0f}} t<extra></extra>"))
        fb.update_layout(barmode="stack", showlegend=False, xaxis_title="Tonnes of residue per year", bargap=0.2)
        fb.update_yaxes(showticklabels=False)
        style_fig(fb, height=190, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fb, width="stretch", config={"displayModeBar": False})
        st.dataframe(alloc, width="stretch", hide_index=True)
        st.download_button("Download this split (CSV)", alloc.to_csv(index=False), "residue_split.csv", "text/csv")

# =====================================================================
# BOTTOM OF PAGE  data, sources and method
# =====================================================================
st.markdown('<div class="sec" style="margin-top:2.4rem">Behind the numbers</div>', unsafe_allow_html=True)
with st.expander("Verified data and sources"):
    st.markdown("""
**Status key:** cited = taken from a source, derived = calculated from cited values, assumption = placeholder.

| Parameter | Value used | Status | Source |
|---|---|---|---|
| Rice straw heating value | 14.6 MJ/kg dry | cited | Springer chapter on rice straw (range 14.08-15.09) |
| Bagasse heating value | 17.9 MJ/kg dry | cited | Taylor & Francis bagasse entry (17.88 +/- 0.48) |
| Pellet price | ₹2.27 per 1000 kcal (about ₹7,900/t) | cited | Ministry of Power benchmark; NTPC paid about ₹7,000/t in 2020 |
| Pellet plant capex | ₹1 crore per t/h | cited | Down To Earth (government scheme assumes ₹35 lakh) |
| Pellet CO2 factors | 1,283 kg avoided, 63-102 kg process per t pellet | cited | NTPC presentation, 2024 |
| Station heat rate / auxiliary | 4,000 kcal/kWh, 10 % | cited | State regulator norms (2025) |
| Biomass power capex | ₹6.38-7.44 crore/MW | cited | CERC generic tariff proposal, 2026 |
| Biomass O&M | ₹54.7 lakh/MW/yr | cited | CERC RE regulations 2024 |
| Biomass power tariff | ₹5.22/kWh (CERC generic proposals ₹9.6-11.6) | cited | PSERC order FY2025-26; Powerline, Aug 2026 |
| Grid emission factor | 0.727 t CO2/MWh | cited | CEA CO2 Baseline Database v20 |
| CBG yield | about 105 kg/t straw | derived | Sangrur plant: 33 t/day from 1.1 lakh t/yr (Mongabay, 2023) |
| CBG price | ₹54/kg (minimum ₹46/kg to 2029) | cited | SATAT / IOCL slab (IEA Task 37 presentation) |
| CBG plant capex | ₹6.5 crore per t/day | cited | CSE national blueprint (₹6-7 crore) |
| Biogas manure (FOM) | 1.75 t per t straw at ₹3/kg, 50 % assumed sold | cited / **assumption** | Mongabay (2023); CSE blueprint |
| 2G ethanol yield | 143 L/t dry straw | cited | ThePrint: 2.1 lakh t straw -> 3 crore L |
| 2G plant capex | over ₹900 crore | cited | IOCL Panipat (PIB, ICIS) |
| Ethanol price | ₹66.61/L | cited | Weighted average ex-mill price ESY 2025-26 (Lok Sabha reply) |
| Biochar yield, carbon retention | 38 %, 45 % | cited / derived | SBJChem 2018; Bioresource Technology 155 (2014) |
| Biochar price and cost | ₹12,000/t, ₹2,500/t residue | **assumption** | No reliable Indian price found; cost from a Vietnam study |
| Pellet operating cost, CBG/ethanol O&M | ₹1,200/t, 5 % of capex | **assumption** | To be replaced with plant data when available |
| Capital charge on plant cost | 10 % discount rate over 20 years | **assumption** | Standard project-finance convention |
| Delivered residue cost | ₹3,000/t | cited | Verbio (Mongabay); IOCL pays ₹1,800/t at depot |

**Cross-checks:** the model's rice-straw ethanol yield reproduces IOCL Panipat, and the biogas lab range (92-280 L CH4/kg VS) gives about 118 kg CH4/t, consistent with the 105 kg CBG/t seen at Sangrur.

**Not yet included:** open-burning baseline (about 1,168 kg CO2 per t straw burnt; Smith et al.), enzyme costs, subsidies (for example capital grants for pellet plants and CBG plants).
""")

with st.expander("How each pathway was calculated"):
    for _, r in df.sort_values("Pathway").iterrows():
        st.markdown(f"**{r['Pathway']}.** {r['Basis']}")
    st.markdown("**Profit** = product sales - plant capital charge - operating cost - residue cost - transport charges (entered by you) - "
                "your other expenses - tax. Tax is charged only on positive profit.")

st.markdown('<div class="foot">Figures are estimates for comparing routes, built from published plant data and prices. '
            'Check the sources above before using them for a decision.</div>', unsafe_allow_html=True)
