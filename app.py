import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import linprog
import streamlit as st

st.set_page_config(page_title="Biorefinery Pathway Optimizer", page_icon="🌾", layout="wide")

# =====================================================================
# DATA LEGEND   [C] cited from a source   [D] derived from cited values
#               [A] assumption (placeholder, replace in Phase III)
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
        power_capex_mw=7.44e7,   # [C] Rs per MW, rice-straw air-cooled (CERC proposal 2026)
    ),
    "Sugarcane bagasse": dict(
        hhv=17.9,          # [C] MJ/kg dry (17.88 +/- 0.48)
        c_frac=0.47,       # [C] ultimate analysis 46.8-48.0 % C
        biochar_yield=0.25,   # [A] literature gives 16 % in a fast free-fall reactor
        ret_c=0.35,        # [A]
        etoh_l=174.0,      # [D] rice-straw yield scaled by carbohydrate fraction (0.51 vs 0.42 L/kg theoretical)
        cbg_kg=None,       # no reliable data yet
        power_capex_mw=6.38e7,   # [C] Rs per MW, biomass Rankine (lower end, CERC proposal 2026)
    ),
}

# ---- constants -------------------------------------------------------
GRID_EF = 0.727          # [C] kg CO2/kWh, CEA CO2 Baseline Database v20, FY2023-24 (RE-adjusted average)
COAL_EF = 94.6           # [C] kg CO2/GJ, IPCC 2006 default (bituminous coal)
CH4_CO2 = 2.75 * 0.95    # [D] kg CO2 per kg CBG (stoichiometry, CBG >= 95 % methane)
ETOH_AVOIDED = 0.66 * 2.31   # [D] kg CO2 per L ethanol (0.66 L petrol-equivalent x 2.31 kg CO2/L)
PELLET_BENCH = 2.27      # [C] Rs per 1000 kcal, MoP benchmark price, northern region, ex-GST
SHR = 4000.0             # [C] kcal/kWh station heat rate, regulator norm
AUX = 0.10               # [C] auxiliary consumption, regulator norm
OM_POWER_MW = 5.47e6     # [C] Rs per MW per year, CERC 2024 O&M norm (FY2024-25)
PLF = 0.80               # [C] plant load factor norm
ETOH_PRICE = 66.61       # [C] Rs/L weighted average ex-mill price, ESY 2025-26 (range 57.97-71.86)
CBG_PRICE = 54.0         # [C] Rs/kg SATAT price slab (minimum guaranteed 46 Rs/kg to 2029)
POWER_TARIFF = 5.22      # [C] Rs/kWh, PSERC approved biomass plant tariff FY2025-26 (CERC generic proposals: 9.6-11.6)


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
        Price=PELLET_BENCH * hhv_kcal,            # [C] benchmark Rs/1000 kcal x calorific value
        Capex=1e7 / (6000 / y),                   # [C] Rs 1 crore per tonne/hour plant (DownToEarth); 6000 h/yr [A]
        Opex=1200.0,                              # [A] power 90-150 kWh/t pellet + labour + maintenance
        Avoided=y * f["hhv"] * COAL_EF,           # [D] cross-check: NTPC reports 1,283 kg CO2 per t pellet
        ProcEmis=y * 102,                         # [C] NTPC: 63-102 kg CO2 per t pellet (upper value)
        Basis="Price = MoP benchmark Rs 2.27 per 1000 kcal x calorific value. Capex Rs 1 crore per t/h. "
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
        ProcEmis=10.0,                            # [A] handling and transport
        Basis="Net kWh = calorific value / heat rate 4000 kcal/kWh x (1 - 10 % auxiliary). "
              "Capex and O&M from CERC norms. Tariff = Punjab regulator FY2025-26. Avoided CO2 = CEA grid factor."))

    # 3. Anaerobic digestion -> CBG (rice straw only) ---------------------
    if f["cbg_kg"]:
        cbg_t_per_day_feed = (1000 / f["cbg_kg"]) * 330
        capex = 6.5e7 / cbg_t_per_day_feed        # [C] Rs 6-7 crore per tonne/day CBG, paddy straw (CSE)
        rows.append(dict(
            Pathway="Anaerobic digestion (CBG)", Goal="Gaseous fuel", Product="CBG", Unit="kg", Yield=f["cbg_kg"],
            Price=CBG_PRICE, Capex=capex, Opex=0.05 * capex,     # O&M 5 % of capex [A]
            Coprod=1.75 * 1000 * 3.0 * 0.5,       # manure: 1.75 t/t straw [C], Rs 3/kg [C], 50 % sold [A]
            Avoided=f["cbg_kg"] * CH4_CO2,
            ProcEmis=40.0,                        # [A]
            Basis="Yield from the Sangrur paddy-straw plant (about 100-110 kg CBG per t). Price = SATAT slab. "
                  "Capex Rs 6.5 crore per t/day CBG. Manure (FOM) income: 1.75 t per t straw x Rs 3/kg x 50 % sold (assumption)."))

    # 4. Slow pyrolysis -> biochar ------------------------------------------
    stored = f["c_frac"] * f["ret_c"] * 44 / 12 * 1000 * 0.80   # kg CO2e per t feed; 80 % permanence [A]
    rows.append(dict(
        Pathway="Slow pyrolysis (biochar)", Goal="Soil / carbon storage", Product="Biochar", Unit="t",
        Yield=f["biochar_yield"], Price=12000.0, Capex=0.0, Opex=2500.0,
        Avoided=stored, ProcEmis=60.0,
        Basis="Yield from literature. Price: no reliable Indian market price (assumption). Cost about Rs 2,500 per t "
              "residue (Vietnam study, low confidence). Stored carbon = C fraction x retained share x 80 % permanence."))

    # 5. Cellulosic (2G) ethanol -----------------------------------------------
    capex = 9e9 / 210000                          # [C] IOCL Panipat: over Rs 900 crore, 2.1 lakh t/yr
    rows.append(dict(
        Pathway="Cellulosic ethanol (2G)", Goal="Liquid fuel", Product="Ethanol", Unit="L", Yield=f["etoh_l"],
        Price=ETOH_PRICE, Capex=capex, Opex=0.05 * capex,        # O&M 5 % of capex [A]; enzymes not included
        Avoided=f["etoh_l"] * ETOH_AVOIDED, ProcEmis=150.0,      # [A]
        Basis="Yield and capex from the IOCL Panipat 2G plant (over Rs 900 crore, 2.1 lakh t straw -> 3 crore L). "
              "Price = weighted average ex-mill ethanol price ESY 2025-26. Enzyme and chemical costs not included."))
    out = pd.DataFrame(rows)
    if "Coprod" not in out:
        out["Coprod"] = 0.0
    out["Coprod"] = out["Coprod"].fillna(0.0)
    return out


# =====================================================================
# SIDEBAR
# =====================================================================
st.title("🌾 Biorefinery Pathway Optimizer")
st.caption("Phase II prototype. Compares conversion pathways for crop residues by profit and CO2 avoided, "
           "using Indian prices and plant data from published sources. Values are tagged cited / derived / assumption.")

with st.sidebar:
    st.header("1. Your inputs")
    feed = st.selectbox("Crop residue", list(FEEDSTOCKS.keys()))
    qty = st.number_input("Quantity (tonnes dry residue / year)", min_value=1.0, value=1000.0, step=100.0)
    feed_cost = st.number_input("Delivered residue cost (Rs per tonne)", min_value=0.0, value=3000.0, step=100.0,
                                help="Collection, transport and storage about Rs 3,000/t (Verbio, via Mongabay). "
                                     "IOCL pays Rs 1,800/t at the depot (ThePrint).")
    goals_all = ["Heat / power", "Gaseous fuel", "Liquid fuel", "Soil / carbon storage"]
    goals = st.multiselect("What do you want to produce?", goals_all, default=goals_all)
    st.header("2. Your priority")
    w = st.slider("CO2 avoided  <-->  Profit", 0, 100, 50, help="0 = only CO2 matters, 100 = only profit matters")
    cap_pct = st.slider("Max share of residue to any one pathway (%)", 20, 100, 40, step=10,
                        help="Used by the Optimization tab. No single route can absorb all residue in practice "
                             "(plant capacity, market demand). Assumption, change it freely.")

    with st.expander("3. Financial assumptions"):
        rate = st.number_input("Discount rate (%)", value=10.0, step=0.5) / 100
        life = st.number_input("Project life (years)", value=20, step=1)
        usd = st.number_input("Rs per US dollar", value=85.0, step=1.0)
        credit = st.number_input("Carbon credit for biochar (US$ per t CO2e)", value=0.0, step=10.0,
                                 help="Voluntary biochar credits have traded at roughly US$100-300 per t CO2e (PelletRates, 2026).")

base = build_pathways(feed)

with st.sidebar:
    with st.expander("4. Edit prices and costs"):
        pr, cx, ox = [], [], []
        for _, r in base.iterrows():
            st.markdown(f"**{r['Pathway']}**")
            pr.append(st.number_input(f"Price (Rs per {r['Unit']})", value=float(round(r["Price"], 2)), key=f"{feed}{r['Pathway']}p"))
            cx.append(st.number_input("Capex (Rs per annual t residue)", value=float(round(r["Capex"], 0)), key=f"{feed}{r['Pathway']}c"))
            ox.append(st.number_input("Operating cost (Rs per t residue)", value=float(round(r["Opex"], 0)), key=f"{feed}{r['Pathway']}o"))
        base["Price"], base["Capex"], base["Opex"] = pr, cx, ox

# =====================================================================
# CALCULATIONS
# =====================================================================
df = base.copy()
df["Revenue (Rs/t)"] = df["Yield"] * df["Price"] + df["Coprod"]
# biochar carbon credit
cred = ((df["Pathway"] == "Slow pyrolysis (biochar)") * df["Avoided"] / 1000 * credit * usd)
df["Revenue (Rs/t)"] = df["Revenue (Rs/t)"] + cred
df["Capital charge (Rs/t)"] = df["Capex"] * crf(rate, life)
df["Profit (Rs/t)"] = df["Revenue (Rs/t)"] - df["Capital charge (Rs/t)"] - df["Opex"] - feed_cost
df["Net CO2 avoided (kg/t)"] = df["Avoided"] - df["ProcEmis"]
df["Total profit (Rs lakh/yr)"] = df["Profit (Rs/t)"] * qty / 1e5
df["Total CO2 avoided (t/yr)"] = df["Net CO2 avoided (kg/t)"] * qty / 1000
df = df[df["Goal"].isin(goals)].reset_index(drop=True)

if df.empty:
    st.warning("Select at least one goal in the sidebar.")
    st.stop()


def norm(s):
    return pd.Series([1.0] * len(s)) if s.max() == s.min() else (s - s.min()) / (s.max() - s.min())


df["Score"] = (w / 100) * norm(df["Profit (Rs/t)"]) + (1 - w / 100) * norm(df["Net CO2 avoided (kg/t)"])
df = df.sort_values("Score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", range(1, len(df) + 1))
best = df.iloc[0]

tab1, tab2 = st.tabs(["Compare pathways", "Optimization"])

with tab1:
    st.subheader(f"Results for {qty:,.0f} t/year of {feed.lower()}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended pathway", best["Pathway"])
    c2.metric("Profit", f"Rs {best['Total profit (Rs lakh/yr)']:,.1f} lakh/yr")
    c3.metric("CO2 avoided", f"{best['Total CO2 avoided (t/yr)']:,.0f} t/yr")
    if best["Profit (Rs/t)"] < 0:
        st.info("Profit is negative with the current inputs for the recommended pathway.")

    st.markdown("#### Ranked pathways")
    show = df[["Rank", "Pathway", "Product", "Yield", "Unit", "Profit (Rs/t)", "Net CO2 avoided (kg/t)", "Score"]].copy()
    show["Yield"] = show["Yield"].round(2)
    show["Profit (Rs/t)"] = show["Profit (Rs/t)"].round(0)
    show["Net CO2 avoided (kg/t)"] = show["Net CO2 avoided (kg/t)"].round(0)
    show["Score"] = show["Score"].round(2)
    st.dataframe(show, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(df, x="Profit (Rs/t)", y="Net CO2 avoided (kg/t)", text="Pathway", color="Pathway",
                         title="Trade-off: profit vs CO2 avoided (per tonne of residue)")
        fig.update_traces(marker=dict(size=14), textposition="top center")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig2 = px.bar(df, x="Pathway", y="Score", color="Pathway", title="Overall score for your priority")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")



# =====================================================================
# OPTIMIZATION  (linear programming with SciPy, epsilon-constraint method)
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
    st.markdown("#### Split the residue across pathways")
    st.caption("A linear program decides how many tonnes go to each pathway. It maximizes profit while meeting a CO2 target "
               "(epsilon-constraint method, Mavrotas, 2009), and traces the profit-vs-CO2 trade-off curve.")
    cap = cap_pct / 100
    if cap * len(df) < 1:
        st.warning("The selected pathways cannot absorb all the residue at this cap. "
                   "Some residue stays unused. Raise the cap or select more goals.")
    a = solve_lp(df, qty, cap, "profit", second="co2")
    b = solve_lp(df, qty, cap, "co2", second="profit")
    if a is None or b is None:
        st.error("The optimization could not be solved. Check your inputs.")
    else:
        co2_a, co2_b = a["co2"], b["co2"]
        if co2_b - co2_a < 1e-6:
            st.success("No trade-off here: the most profitable allocation is also the best for CO2.")
            chosen = a
        else:
            pts = []
            for k in range(9):
                eps = co2_a + (co2_b - co2_a) * k / 8
                r = solve_lp(df, qty, cap, "profit", co2_min=eps - 1e-6)
                if r:
                    pts.append(r)
            t = st.slider("Choose a point on the trade-off curve", 0, 100, 50,
                          help="0 = maximum profit, 100 = maximum CO2 avoided")
            eps_sel = co2_a + (co2_b - co2_a) * t / 100
            chosen = solve_lp(df, qty, cap, "profit", co2_min=eps_sel - 1e-6)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=[p["profit"] / 1e5 for p in pts], y=[p["co2"] / 1000 for p in pts],
                                      mode="lines+markers", name="Pareto front"))
            fig3.add_trace(go.Scatter(x=[chosen["profit"] / 1e5], y=[chosen["co2"] / 1000], mode="markers",
                                      marker=dict(size=16, color="red"), name="Your choice"))
            fig3.update_layout(title="Trade-off curve: profit vs CO2 avoided", xaxis_title="Profit (Rs lakh/yr)",
                               yaxis_title="CO2 avoided (t/yr)")
            st.plotly_chart(fig3, width="stretch")

        alloc = pd.DataFrame({
            "Pathway": df["Pathway"],
            "Residue allocated (t/yr)": [round(v, 1) for v in chosen["x"]],
            "Share (%)": [round(100 * v / qty, 1) for v in chosen["x"]],
            "Profit (Rs lakh/yr)": [round(df.loc[i, "Profit (Rs/t)"] * chosen["x"][i] / 1e5, 2) for i in range(len(df))],
            "CO2 avoided (t/yr)": [round(df.loc[i, "Net CO2 avoided (kg/t)"] * chosen["x"][i] / 1000, 1) for i in range(len(df))],
        })
        alloc = alloc[alloc["Residue allocated (t/yr)"] > 0.05].reset_index(drop=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total profit", f"Rs {chosen['profit'] / 1e5:,.1f} lakh/yr")
        m2.metric("Total CO2 avoided", f"{chosen['co2'] / 1000:,.0f} t/yr")
        m3.metric("Residue used", f"{sum(chosen['x']):,.0f} of {qty:,.0f} t")
        if sum(chosen["x"]) < qty - 1:
            st.caption(f"{qty - sum(chosen['x']):,.0f} t of residue is left unprocessed: routes that lose money are skipped "
                       "unless they are needed to reach the CO2 target, or the cap on each pathway is reached.")
        st.markdown("##### Recommended allocation")
        st.dataframe(alloc, width="stretch", hide_index=True)
        figb = px.bar(alloc, x="Pathway", y="Residue allocated (t/yr)", color="Pathway",
                      title="How the residue is split")
        figb.update_layout(showlegend=False)
        st.plotly_chart(figb, width="stretch")
        st.download_button("Download allocation (CSV)", alloc.to_csv(index=False), "allocation.csv", "text/csv")

# =====================================================================
# DATA BASIS
# =====================================================================
with st.expander("How each pathway was calculated"):
    st.dataframe(df[["Pathway", "Basis"]], width="stretch", hide_index=True)

with st.expander("Verified data and sources"):
    st.markdown("""
**Status key:** cited = taken from a source, derived = calculated from cited values, assumption = placeholder.

| Parameter | Value used | Status | Source |
|---|---|---|---|
| Rice straw heating value | 14.6 MJ/kg dry | cited | Springer chapter on rice straw (range 14.08-15.09) |
| Bagasse heating value | 17.9 MJ/kg dry | cited | Taylor & Francis bagasse entry (17.88 +/- 0.48) |
| Pellet price | Rs 2.27 per 1000 kcal (about Rs 7,900/t) | cited | Ministry of Power benchmark; NTPC paid about Rs 7,000/t in 2020 |
| Pellet plant capex | Rs 1 crore per t/h | cited | Down To Earth (government scheme assumes Rs 35 lakh) |
| Pellet CO2 factors | 1,283 kg avoided, 63-102 kg process per t pellet | cited | NTPC presentation, 2024 |
| Station heat rate / auxiliary | 4,000 kcal/kWh, 10 % | cited | State regulator norms (2025) |
| Biomass power capex | Rs 6.38-7.44 crore/MW | cited | CERC generic tariff proposal, 2026 |
| Biomass O&M | Rs 54.7 lakh/MW/yr | cited | CERC RE regulations 2024 |
| Biomass power tariff | Rs 5.22/kWh (CERC generic proposals Rs 9.6-11.6) | cited | PSERC order FY2025-26; Powerline, Aug 2026 |
| Grid emission factor | 0.727 t CO2/MWh | cited | CEA CO2 Baseline Database v20 |
| CBG yield | about 105 kg/t straw | derived | Sangrur plant: 33 t/day from 1.1 lakh t/yr (Mongabay, 2023) |
| CBG price | Rs 54/kg (minimum Rs 46/kg to 2029) | cited | SATAT / IOCL slab (IEA Task 37 presentation) |
| CBG plant capex | Rs 6.5 crore per t/day | cited | CSE national blueprint (Rs 6-7 crore) |
| Biogas manure (FOM) | 1.75 t per t straw at Rs 3/kg, 50 % assumed sold | cited / **assumption** | Mongabay (2023); CSE blueprint |
| 2G ethanol yield | 143 L/t dry straw | cited | ThePrint: 2.1 lakh t straw -> 3 crore L |
| 2G plant capex | over Rs 900 crore | cited | IOCL Panipat (PIB, ICIS) |
| Ethanol price | Rs 66.61/L | cited | Weighted average ex-mill price ESY 2025-26 (Lok Sabha reply) |
| Biochar yield, carbon retention | 38 %, 45 % | cited / derived | SBJChem 2018; Bioresource Technology 155 (2014) |
| Biochar price and cost | Rs 12,000/t, Rs 2,500/t residue | **assumption** | No reliable Indian price found; cost from a Vietnam study |
| Pellet operating cost, CBG/ethanol O&M | Rs 1,200/t, 5 % of capex | **assumption** | To be replaced with plant data in Phase III |
| Delivered residue cost | Rs 3,000/t | cited | Verbio (Mongabay); IOCL pays Rs 1,800/t at depot |

**Cross-checks:** the model's rice-straw ethanol yield reproduces IOCL Panipat, and the biogas lab range (92-280 L CH4/kg VS) gives about 118 kg CH4/t, consistent with the 105 kg CBG/t seen at Sangrur.

**Not yet included:** open-burning baseline (about 1,168 kg CO2 per t straw burnt; Smith et al.), enzyme costs, subsidies (for example capital grants for pellet plants and CBG plants).
""")
