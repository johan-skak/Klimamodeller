import streamlit as st
import matplotlib.pyplot as plt
from main import main

# Formatting of selected options in multiselect
st.markdown("""
    <style>
        /* --- Highlight of selected options --- */
        span[data-baseweb="tag"] {
            background-color: #80b080 !important;
        }
            
        /* Change the max width of the main content area */
        .block-container {
            max-width: 90%;
            width: 1500px;
        }
    </style>
""", unsafe_allow_html=True)


tabs_html = """
    <style>
        /* Scrollable container */
        div[aria-label="Vælg figur:"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        }

        /* Hide default radio dots */
        div[aria-label="Vælg figur:"] label > div:first-child {
            display: none !important;
        }

        /* Tabs */
        div[aria-label="Vælg figur:"] label {
            display: inline-block !important;
            background-color: transparent !important;
            border-bottom: 2px solid #31333f1A; /* Streamlit default border color with transparency */
            margin-right: 0;
            margin-bottom: .5rem;
            cursor: pointer;
            white-space: nowrap;
            transition: border-color 0.2s;
        }

        /* Tab text */
        div[aria-label="Vælg figur:"] label div {
            transition: color 0.2s;
            font-size: 14px;
            color: #31333f; /* Default Streamlit text color */
            padding-left: 3.5px;
            padding-right: 3.5px;
        }

        /* First tab text */
        div[aria-label="Vælg figur:"] label:first-child div {
            padding-left: 0 !important;
        }

        /* Hover effect */
        div[aria-label="Vælg figur:"] label div:hover {
            color: #ff4b4b; /* Streamlit red */
        }

        /* Active tab */
        div[aria-label="Vælg figur:"] label:has(input:checked) {
            border-bottom: 2px solid #ff4b4b !important;
        }

        /* Active tab text */
        div[aria-label="Vælg figur:"] label:has(input:checked) div {
            color: #ff4b4b !important; /* Streamlit red */
        }
    </style>
"""


@st.cache_data
def run(params, config):
    return main(config | {"output_dir": "Results"}, params, app=True) # output_dir is ignored in app mode but must be a string 

def plot_in_tabs(axes_funcs, hash_code):
    figs, titles = make_plots_and_titles(axes_funcs, hash_code)
    if not figs:
        st.info("Ingen figurer at vise. Prøv at ændre parametre eller konfiguration.")
        return
    st.markdown(tabs_html, unsafe_allow_html=True)
    choice = st.radio("Vælg figur:", titles, horizontal=True, label_visibility="collapsed")
    st.pyplot(figs[titles.index(choice)])

@st.cache_data # Hash_code is only for hashing uniquely
def make_plots_and_titles(_axes_funcs, hash_code):
    figs = []
    titles = []
    for ax_func in _axes_funcs:
        fig, ax = plt.subplots()
        ax_func(ax)
        figs.append(fig)
        titles.append(ax.get_title())
    return figs, titles

def make_summary(summaries):
    if not summaries: return ""
    summary = "\n\n".join(summaries)
    return summary

def set_keyed_inputs(defaults_dict):
    for k, v in defaults_dict.items():
        if k not in st.session_state: raise ValueError(f"Key {k} not in session_state")
        st.session_state[k] = v

DEFAULT_PARAMS = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288, SD=250, S0=1365, S1=None, F=4.0)
DEFAULT_CONFIG = dict(years=1000, ctrl_years=None, dt_years=1.0, nx=200, modes=[])

DEFAULT_SEVA_PARAMS = DEFAULT_PARAMS | dict(F=0.0, SD=20) # No forcing in seasonal variation mode
DEFAULT_SEVA_CONFIG = DEFAULT_CONFIG | dict(years=50, dt_years=1/24, modes=["SeasonalVariation"])

DEFAULT_SEADEP_PARAMS = DEFAULT_PARAMS | dict(F=0.0, SD=None)
DEFAULT_SEADEP_CONFIG = DEFAULT_CONFIG | dict(modes=["VariableSeaDepth"])

DEFAULT_SEVA_SEADEP_PARAMS = DEFAULT_SEVA_PARAMS | dict(SD=None)
DEFAULT_SEVA_SEADEP_CONFIG = DEFAULT_SEVA_CONFIG | dict(modes=["SeasonalVariation", "VariableSeaDepth"])

# Initialize with default values each time the script is rerun
params = DEFAULT_PARAMS.copy()
config = DEFAULT_CONFIG.copy()

# --- Begin Streamlit app ---
st.title("Energibalance model af Jordens klima")

# Top buttons in one row
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
with col_btn1:
    # --- Default button ---
    if st.button("Standardtilstand", icon="🔄"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS)
        set_keyed_inputs(DEFAULT_CONFIG)
with col_btn2:
    # --- Seasonal Variation button ---
    if st.button("Sæsonvariation", icon="🌱"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_SEVA_PARAMS)
        set_keyed_inputs(DEFAULT_SEVA_CONFIG)
with col_btn3:
    # --- Variable Sea Depth button ---
    if st.button("Variabel havdybde", icon="🌊"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_SEADEP_PARAMS)
        set_keyed_inputs(DEFAULT_SEADEP_CONFIG)
with col_btn4:
    # --- Seasonal Variation + Variable Sea Depth button ---
    if st.button("🌱🌊 Sæsonvariation + Variabel havdybde"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_SEVA_SEADEP_PARAMS)
        set_keyed_inputs(DEFAULT_SEVA_SEADEP_CONFIG)


with st.form("input_form"):
    col1, col2 = st.columns(2) # Spacing column in the middle
    with col1:
        with st.expander("🧮 Parametre"):
            st.header("Parametre")
            # Overwrite params dict with user input when form is submitted
            F  = st.number_input(r"F: Ekstra strålingspåvirkning (W/m²)", value=DEFAULT_PARAMS["F"], key="F", step=1.0)
            SD = st.number_input("SD: Varmekapacitet i meter havdybde (m)", value=DEFAULT_PARAMS["SD"], key="SD", step=10)
            D0 = st.number_input("D0: Diffusionskoefficient (m²/s)", value=DEFAULT_PARAMS["D0"], key="D0", step=0.1)
            T0 = st.number_input("T0: Initial temperatur (K)", value=DEFAULT_PARAMS["T0"], key="T0", step=10)
            S0 = st.number_input("S0: Solindstråling under kontrolperiode (W/m²)", value=DEFAULT_PARAMS["S0"], key="S0", step=50)
            S1 = st.number_input("S1: Solindstråling efter kontrolperiode (W/m²) (lad stå tom for ingen ændring)", value=DEFAULT_PARAMS["S1"], key="S1", step=50)
            k1 = st.number_input("k1: Temperatursensitivitet for isdannelse (K⁻¹)", value=DEFAULT_PARAMS["k1"], key="k1", step=0.01)
            k2 = st.number_input("k2: Temperatursensitivitet for diffusivitet (K⁻¹)", value=DEFAULT_PARAMS["k2"], key="k2", step=0.005)
            k3 = st.number_input("k3: Feedbackstyrke af drivhuseffekten", value=DEFAULT_PARAMS["k3"], key="k3", step=0.1)
            input_dict = dict(F=F, SD=SD, D0=D0, T0=T0, S0=S0, S1=S1, k1=k1, k2=k2, k3=k3)
            for key, value in input_dict.items():
                if value is not None: params[key] = value

    with col2:
        with st.expander("⚙️ Konfiguration"):
            st.header("Konfiguration")
            # Overwrite config dict with user input when form is submitted
            years       = st.number_input("Simuleringstid (år)", value=DEFAULT_CONFIG["years"], key="years", step=50, min_value=1, max_value=1000)
            ctrl_years  = st.number_input("Kontrolperiode (år) (lad stå tom for halvdelen af simuleringstiden)", value=DEFAULT_CONFIG["ctrl_years"], key="ctrl_years", step=50, min_value=0, max_value=1000)
            dt_years    = st.number_input("Tidsskridt (år)", value=DEFAULT_CONFIG["dt_years"], key="dt_years", step=0.01, min_value=0.01, max_value=10.0)
            nx          = st.number_input("Antal gitterpunkter", value=DEFAULT_CONFIG["nx"], key="nx", step=100, min_value=10, max_value=1000)
            modes       = st.multiselect("Modeller", options=["SeasonalVariation", "VariableSeaDepth"], default=DEFAULT_CONFIG["modes"], key="modes")
            input_dict  = dict(years=years, ctrl_years=ctrl_years, dt_years=dt_years, nx=nx, modes=modes)
            for key, value in input_dict.items():
                if value is not None:
                    config[key] = value

    submitted = st.form_submit_button("▶️ Kør simulation med opdaterede parametre og konfiguration", width="stretch")

# Run simulation and show outputs
axes_funcs, summaries = run(params, config)
col_out1, col_out2 = st.columns(2, border=True)
with col_out1:
    plot_in_tabs(axes_funcs, summaries)
with col_out2:
    st.header("Sammenfatning af simulation")
    if summaries:
        st.text(make_summary(summaries))
