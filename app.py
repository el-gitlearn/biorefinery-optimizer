import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Biorefinery Pathway Optimizer", page_icon="🌾", layout="wide")

# =====================================================================
# 1. FEEDSTOCK DATA  (per tonne of DRY feedstock)
#    status: cited = taken from a paper | derived = calculated from cited
#            values | assumption = placeholder to be replaced in Phase 3
# =====================================================================
FEEDSTOCKS = {
    "Rice straw": {
        "hhv_gj_t": 15.0,          # assumption (typical 14-16 GJ/t, verify)
        "carbon_frac": 0.38,       # assumption (verify with ultimate analysis)
        "vs_frac": 0.86,           # derived: VS 80.82 / TS 93.7 (PMC9055357)
        "ethanol_theoretical_l_kg": 0.42,   # cited: IJTech 2015 (cellulose 38.6 %, hemicellulose 19.7 %)
        "biochar_yield": 0.38,     # interpolated: 33.9-49.9 wt% over 700-300 C (SBJChem 2018 paper, see Sources)
        "ch4_l_per_kg_vs": 190,    # cited range 92-280 L/kg VS (review, see Sources); mid value
    },
    "Sugarcane bagasse": {
        "hhv_gj_t": 17.9,          # cited: 17.88 MJ/kg dry (range 17-19.9)
        "carbon_frac": 0.47,       # cited: ultimate analysis 46.8-48.0 % C
        "vs_frac": None,
        "ethanol_theoretical_l_kg": 0.51,   # derived: scaled from rice straw by carbohydrate fraction (about 71 % vs 58.3 %)
        "biochar_yield": 0.25,     # assumption (lit. gives 16 % in a fast free-fall reactor)
        "ch4_l_per_kg_vs": None,
    },
}

# Constants
GRID_EF = 0.727          # kg CO2 / kWh, CEA CO2 Baseline Database v20 (FY 2023-24, RE-adjusted average)
COAL_EF = 94.6           # kg CO2 / GJ, IPCC default (bituminous coal)
CH4_CO2 = 1.96           # kg CO2 per m3 CH4 burned (stoichiometry)
ETOH_AVOIDED = 0.66 * 2.31   # kg CO2 per L ethanol (0.66 L petrol-equivalent x 2.31 kg CO2/L)
BIOCHAR_RETAINED_C = 0.45    # 45 % of straw carbon retained in biochar (slow pyrolysis of rice straw, Bioresour. Technol. 155, 2014)
BIOCHAR_PERMANENCE = 0.80    # assumption: share that stays stable in soil


def build_pathways(feed):
    f = FEEDSTOCKS[feed]
    rows = []

    # Pelletization (heat fuel)
    pellet_yield = 0.90   # t pellets / t feed (assumption)
    rows.append(dict(
        Pathway="Pelletization", Goal="Heat / power", Product="Pellets", Unit="t",
        Yield=pellet_yield, Price=7500.0, ProcCost=1500.0, ProcEmis=70.0,
        Avoided=pellet_yield * f["hhv_gj_t"] * COAL_EF,
        Basis="Yield, price, cost: assumptions. Avoided CO2 = pellet energy x coal EF (IPCC)"))

    # Direct combustion -> electricity
    eff = 0.25
    kwh = f["hhv_gj_t"] * 1000 / 3.6 * eff
    rows.append(dict(
        Pathway="Combustion for power", Goal="Heat / power", Product="Electricity", Unit="kWh",
        Yield=kwh, Price=5.0, ProcCost=1800.0, ProcEmis=20.0,
        Avoided=kwh * GRID_EF,
        Basis="Energy from HHV x 25 % efficiency (assumption). Avoided CO2 = grid EF (CEA v20)"))

    # Anaerobic digestion (only where data is available)
    if f["ch4_l_per_kg_vs"]:
        m3 = f["vs_frac"] * 1000 * f["ch4_l_per_kg_vs"] / 1000
        rows.append(dict(
            Pathway="Anaerobic digestion (biomethane)", Goal="Gaseous fuel", Product="Methane", Unit="m3",
            Yield=m3, Price=35.0, ProcCost=2500.0, ProcEmis=40.0,
            Avoided=m3 * CH4_CO2,
            Basis="CH4 yield 92-280 L/kg VS (review), mid value used. Price, cost: assumptions"))

    # Slow pyrolysis -> biochar
    biochar_t = f["biochar_yield"]
    c_stored = f["carbon_frac"] * BIOCHAR_RETAINED_C * 44 / 12 * 1000 * BIOCHAR_PERMANENCE
    rows.append(dict(
        Pathway="Slow pyrolysis (biochar)", Goal="Soil / carbon storage", Product="Biochar", Unit="t",
        Yield=biochar_t, Price=12000.0, ProcCost=3000.0, ProcEmis=60.0,
        Avoided=c_stored,
        Basis="Biochar yield from literature. Carbon stored = C fraction x 45 % retained x 80 % permanence (assumption)"))

    # Cellulosic ethanol
    overall_eff = 0.55
    litres = f["ethanol_theoretical_l_kg"] * 1000 * overall_eff
    rows.append(dict(
        Pathway="Cellulosic ethanol", Goal="Liquid fuel", Product="Ethanol", Unit="L",
        Yield=litres, Price=60.0, ProcCost=6500.0, ProcEmis=150.0,
        Avoided=litres * ETOH_AVOIDED,
        Basis="Theoretical yield x 55 % overall efficiency (assumption). Price, cost: assumptions"))

    return pd.DataFrame(rows)


# =====================================================================
# 2. SIDEBAR  (user inputs)
# =====================================================================
st.title("🌾 Biorefinery Pathway Optimizer")
st.caption("Prototype (Phase II). Compares conversion pathways for crop residues by profit and CO2 avoided. "
           "Yields come from published studies; prices and process costs are editable assumptions.")

with st.sidebar:
    st.header("1. Your inputs")
    feed = st.selectbox("Crop residue", list(FEEDSTOCKS.keys()))
    qty = st.number_input("Quantity (tonnes dry residue / year)", min_value=1.0, value=1000.0, step=100.0)
    feed_cost = st.number_input("Residue cost (Rs per tonne)", min_value=0.0, value=1500.0, step=100.0)
    goals_all = ["Heat / power", "Gaseous fuel", "Liquid fuel", "Soil / carbon storage"]
    goals = st.multiselect("What do you want to produce?", goals_all, default=goals_all)
    st.header("2. Your priority")
    w = st.slider("Profit  <-->  CO2 avoided", 0, 100, 50,
                  help="0 = care only about CO2, 100 = care only about profit")

base = build_pathways(feed)

with st.sidebar:
    with st.expander("3. Edit assumptions (prices and costs)"):
        edited = []
        for _, r in base.iterrows():
            st.markdown(f"**{r['Pathway']}**")
            price = st.number_input(f"Price (Rs per {r['Unit']})", value=float(r["Price"]), key=f"{feed}_{r['Pathway']}_p")
            cost = st.number_input("Processing cost (Rs per t residue)", value=float(r["ProcCost"]), key=f"{feed}_{r['Pathway']}_c")
            emis = st.number_input("Process emissions (kg CO2 per t residue)", value=float(r["ProcEmis"]), key=f"{feed}_{r['Pathway']}_e")
            edited.append((price, cost, emis))
        base["Price"] = [e[0] for e in edited]
        base["ProcCost"] = [e[1] for e in edited]
        base["ProcEmis"] = [e[2] for e in edited]

# =====================================================================
# 3. CALCULATIONS
# =====================================================================
df = base.copy()
df["Revenue (Rs/t)"] = df["Yield"] * df["Price"]
df["Profit (Rs/t)"] = df["Revenue (Rs/t)"] - df["ProcCost"] - feed_cost
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

# =====================================================================
# 4. RESULTS
# =====================================================================
st.subheader(f"Results for {qty:,.0f} t/year of {feed.lower()}")
c1, c2, c3 = st.columns(3)
c1.metric("Recommended pathway", best["Pathway"])
c2.metric("Profit", f"Rs {best['Total profit (Rs lakh/yr)']:,.1f} lakh/yr")
c3.metric("CO2 avoided", f"{best['Total CO2 avoided (t/yr)']:,.0f} t/yr")

if best["Profit (Rs/t)"] < 0:
    st.info("Profit is negative with the current assumptions. Try changing prices or costs in the sidebar.")

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
# 5. DATA BASIS
# =====================================================================
with st.expander("How each pathway was calculated (data basis)"):
    st.dataframe(df[["Pathway", "Basis"]], width="stretch", hide_index=True)
    st.markdown(
        "**Status of the numbers:** yields are taken or derived from the studies below. "
        "Prices, processing costs, efficiencies marked as assumptions are placeholders and will be replaced "
        "with cited techno-economic data in Phase III.")

with st.expander("Sources used so far"):
    st.markdown("""
Full APA references will be formatted in the report. Links below are the pages the numbers were taken from.

- **Rice straw solids (TS/VS):** https://pmc.ncbi.nlm.nih.gov/articles/PMC9055357
- **Rice straw methane yield (range 92-280 L/kg VS):** review at https://core.ac.uk/download/322792246.pdf
- **Rice straw biochar yield (33.9-49.9 wt%):** https://doi.org/10.48141/SBJCHEM.v26.n26.2018.22
- **Carbon retained in biochar (45 %):** Slow pyrolysis of rice straw, Bioresource Technology, 155 (2014): https://www.pubmed.gov/24423650/
- **Rice straw theoretical ethanol yield (0.42 L/kg):** https://ijtech.eng.ui.ac.id/download/article/1326
- **Bioethanol from rice straw overview:** Binod, P., et al. (2010). *Bioresource Technology, 101*(13), 4767-4774.
- **Bagasse heating value (17.88 MJ/kg dry):** https://taylorandfrancis.com/knowledge/Engineering_and_technology/Power_%26_energy/Bagasse
- **Bagasse carbon content and pyrolysis yields:** https://fjet.fudutsinma.edu.ng/index.php/fjet/article/download/40/37/75
- **India grid emission factor (0.727 t CO2/MWh):** CEA CO2 Baseline Database v20 (Dec 2024), https://cea.nic.in
- **Coal emission factor:** IPCC (2006) Guidelines for National Greenhouse Gas Inventories
- **Biomass composition database:** TNO Phyllis2, https://phyllis.nl/
""")
