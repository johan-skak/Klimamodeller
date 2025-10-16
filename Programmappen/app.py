import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import re, io, os
from main import main
from outputs import generate_outputs_data, print_simulation_info

class ButtonGroup:
    def __init__(self, specials=None, funcs=None):
        self.specials = specials if isinstance(specials, list) else ( [specials] if specials else [] )
        self.funcs = funcs if isinstance(funcs, list) else ( [funcs] if funcs else [] )
        if len(self.specials) != len(self.funcs):
            raise ValueError("specials and funcs must have the same length")
        self.clicked = None

    def button(self, label, special=None, **kwargs):
        """Creates a Streamlit button and remembers if it was clicked."""
        def run_first(inFunc=None):
            """Run the inFunc first (if any), then the special funcs that match the special argument (if any)."""
            if inFunc: inFunc()
            # Call the functions with True/False depending on whether the special button was clicked
            for _special, func in zip(self.specials, self.funcs):
                func(_special == special)
        
        # Callback runs run_first then any on_click function given in kwargs
        inFunc = kwargs.pop("on_click", None)
        kwargs["on_click"] = lambda: run_first(inFunc)

        if st.button(label, **kwargs):
            self.clicked = label
            # st.session_state["last_clicked"] = label

        return self.clicked == label

st.html("<style> .stElementContainer:has(.no-gap) {margin-top: -1em; margin-bottom: -1em;} </style>")
tabs_html = """
    <style>
        /* Scrollable container */
        div[aria-label="Vælg figur:"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        margin-bottom: 1rem;
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
    return main(config | {"output_dir": "Results"}, params, app=True) # output_dir is speciald in app mode but must be a string 

def plot_in_tabs(axes_funcs, hash_code):
    figs, titles = make_plots_and_titles(axes_funcs, hash_code)
    if not figs:
        st.info("Ingen figurer at vise. Prøv at ændre parametre eller opsætning.")
        return
    
    # Remember last choice if possible
    index = titles.index(st.session_state.get("choice", titles[0])) if st.session_state.get("choice") in titles else 0
    st.html(tabs_html)
    st.radio("Vælg figur:", titles, horizontal=True, label_visibility="collapsed", key="choice", index=index)
    fig = figs[titles.index(st.session_state["choice"])]
    size = 6
    if not isinstance(fig, plt.Figure):
        style = """<style>.animation {width: 100%} .animation img {width: 100%; margin-top: -15px; margin-bottom: -20px}
                    .anim-controls > input {width: 100% !important} body {margin: 0}</style>"""
        html = call_animate_on_earth(fig, size, hash_code) # Call the function to get the animation
        st.html("<style>.stElementContainer:has(iframe) {aspect-ratio: 1 / 1.25; height: auto; margin-top: -1em;} div[aria-label='Vælg figur:'] {margin: 0;}</style>")
        components.html(style+html)
    else:
        # fig.set_size_inches(size, size)
        st.pyplot(figs[titles.index(st.session_state["choice"])])

@st.cache_data
def call_animate_on_earth(_func, size, hash_code):
    return _func(size)

# @st.cache_data # Hash_code is only for hashing uniquely
def make_plots_and_titles(_axes_funcs, hash_code):
    figs = []
    titles = []
    for ax_func in _axes_funcs:
        fig, ax = plt.subplots()
        ani_func = ax_func(ax)
        if ani_func is not None:
            figs.append(ani_func)
        else:
            figs.append(fig)
        titles.append(ax.get_title())
    return figs, titles

def make_summary(summaries):
    if not summaries: return ""
    summary = "\n\n".join(summaries)
    return ansi_to_html(summary)

def ansi_to_html(s):
    """Convert basic ANSI colors to HTML spans."""
    colors = {
        "31": "#ff5100", # orange-red
        "32": "green",
        "33": "yellow",
        "34": "#00aeff", # ice-blue
        "35": "magenta",
        "36": "cyan",
        "90": "gray",
    }
    style = """
        <style>
        pre {
                white-space: pre-wrap;       /* preserve spacing + wrap long lines */
                word-break: break-word;      /* allow breaking inside long words */
                font-size: 0.95rem;
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 0.5rem;
                background: #1ea8493d;  /* Light green background */
                border-radius: 0.5rem;
                overflow-x: auto; /* Enable horizontal scrolling if needed */
                color: #31333f;              /* Streamlit default text color */
            }
        </style>
    """
    color_style = ""
    for code, color in colors.items():
        color_style += f"._{code} {{color: {color};}}\n"
    style += f"<style>\n{color_style}</style>"

    # Replace \033[<n>m with <span style="color:...">
    for code, color in colors.items():
        s = re.sub(fr"\033\[{code}m", f"<span class='_{code}'>", s)
    # Reset code
    s = re.sub(r"\033\[0m", "</span>", s)
    # Warn if any unhandled codes remain
    if re.search(r"\033\[\d+m", s):
        st.info("Warning: Unhandled ANSI codes remain in summary.")
    return f"{style}<pre>{format_terminal_output(s)}</pre>"

def format_terminal_output(text):
    lines = [l.rstrip() for l in text.splitlines()]
    html_lines = []
    block = []

    def make_space(string, pattern, space):
        return re.sub(fr"({pattern})",
                    fr"<span style='display:inline-block; min-width:{space}em; text-align:right;'>\1</span>", 
                    string)

    def estimate_pixel_width(s: str, px_per_char=8, px_per_wide=12, px_per_medium_wide=10, px_per_narrow=5):
        wide = sum(1 for c in s if c in "W@")
        medium_wide = sum(1 for c in s if c in "mM")
        narrow = sum(1 for c in s if c in "iIl.,;:'|!1 ()-°")
        normal = len(s) - wide - narrow - medium_wide
        return wide * px_per_wide + medium_wide * px_per_medium_wide + narrow * px_per_narrow + normal * px_per_char + 5

    def flush_block(block):
        """Render one logical block"""
        if not block:
            return ""
        # Check if all lines have exactly one colon
        if all(line.count(":") == 1 for line in block):
            # Compute maximal widths of each column over all lines
            max_key_px = max(estimate_pixel_width(line.split(":", 1)[0]) for line in block)
            max_val_px = max(estimate_pixel_width(re.sub(r'<[^>]+>', '', line.split(":", 1)[1])) for line in block)
            html = f"""<div class='block' style='grid-template-columns: minmax({int(max_key_px*0.6)}px, max-content) minmax({int(max_val_px*0.5)}px, max-content); /* Makes the columns wrap if needed but only to two lines */'>"""
            for line in block:
                key, val = line.split(":", 1)
                # Highlight numbers with inline-block spans (optional)
                val_html = make_space(val, r"\s*[+-]?\d+[.,]\d{2}(?!(\d|[a-z]))", 2.9) # Numbers like 123.45 or -0.52
                val_html = make_space(val_html, r"\s*[+-]?\d+[.,]\d{1}(?!(\d|[a-z]))", 2.3) # Numbers like 123.4 or -0.5
                val_html = make_space(val_html, r"[A-Z][a-z]{2}", 1.8) # 3-letter month abbreviations
                html += f"""
                <div class='key'>{key.strip()}<b>:</b></div>
                <div class='value'>{val_html}</div>
                """
            html += "</div><br>"
            return html
        else:
            # Otherwise, flat text block
            return "<div style='text-wrap: nowrap;'>" + "<br>".join(block) + "</div><br>"

    # Group lines
    for line in lines + [""]:
        if not line.strip(): # Empty line indicates new block
            html_lines.append(flush_block(block))
            block = []
        else:
            block.append(line)
    html_lines[-1] = html_lines[-1].rstrip("<br>") # Remove very last <br>
    
    block_style = """<style>
        .block {
            display:grid;
            line-height: 1.2;
        }
        .block .key { 
            grid-column:1;
            text-align:left;
            text-wrap: balance;
            align-self: start;
        }
        .block .value {
            grid-column:2;
            text-align:right;
            align-self: end;
        }
        .block > div {
            margin-bottom:0.6em;
            white-space:normal;
        }
        </style>"""
    # Combine all
    return block_style + ''.join(html_lines)

def set_keyed_inputs(defaults_dict):
    for k, v in defaults_dict.items():
        st.session_state[k] = v # Update forms and create keys if they don't exist yet (toggle off)

def make_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# Initialize expand state to false
if "expand" not in st.session_state:
    st.session_state.expand = False
if "show_all_params" not in st.session_state:
    st.session_state.show_all_params = False
# Function to toggle expand state
def show_params_expander():
    if st.session_state.show_all_params:
        st.session_state.expand = True
    else:
        st.session_state.expand = False

# Initialize run_away state to false
if "run_away" not in st.session_state:
    st.session_state.run_away = False
# Function to set run_away
def set_run_away(Bool):
    st.session_state.run_away = Bool
# Create ButtonGroup instance
btns = ButtonGroup("run_away", set_run_away)


DEFAULT_PARAMS = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288, SD=250, S0=1365, S1=None, F=4.0)
DEFAULT_CONFIG = dict(years=1000, ctrl_years=None, dt_years=1.0, nx=200, modes=[])

# Initialize with default values
st.session_state.params = DEFAULT_PARAMS.copy()
st.session_state.config = DEFAULT_CONFIG.copy()

# Set page config
st.set_page_config(page_title="Energibalancemodel af Jordens klima", page_icon="🌍",)
st.html("""
    <style>            
        /* Change the max width of the main content area */
        .block-container {
            max-width: 90%;
            width: 1500px;
        }
    </style>
""")

# Sidebar for more presets and info
st.html("<style> div.stHorizontalBlock:has(.slim-container) {gap: 0.3em} </style>")
with st.sidebar:
    st.markdown("---")
    st.html("<style> h2 {padding: 0 !important;} </style>")
    st.header("Forudindstillinger")
    st.html("<i>Nogle forudindstillinger for simple eksperimenter.</i>")
    if btns.button("Standardtilstand", icon="🔄"):
        set_keyed_inputs(DEFAULT_PARAMS)
        set_keyed_inputs(DEFAULT_CONFIG)
    if btns.button("Nul-diffusion", icon=":material/mode_fan_off:"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(D0 = 0.0, F=0.0))  # No forcing in seasonal variation mode
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile"
    if btns.button("Havdybde 2000 m", icon="🌊"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=2000))
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Global Mean Surface Temperature"
    if btns.button("Havdybde 20 m", icon="🪨"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=20))
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Global Mean Surface Temperature"
    if btns.button("Snebold-Jorden", icon="❄️"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, T0=245))  # No forcing and no sea depth
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile"
    if btns.button("Løbsk drivhuseffekt", special="run_away", icon="🔥"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=20, k3=1.1))  # Strong forcing and strong feedback
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile"
    if st.session_state.run_away: # Show input field if the last button click was runaway
        col_run_away1, col_run_away2 = st.columns([1, 2])
        col_run_away1.html("<div class='slim-container'>Indstil k3:</div>")
        col_run_away2.number_input("Label", label_visibility="collapsed", value=1.1, key="k3_runaway", step=0.1)
        st.session_state["k3"] = st.session_state["k3_runaway"]


# --- Header and download buttons ---
st.html("""<style>
            .stHorizontalBlock:has(.downloadButton) {/* The header with title and download buttons */
            display: flex;
            flex-wrap: wrap;         /* allow multiple lines */
            justify-content: space-between;
            align-items: flex-end;   /* align bottoms of wrapped lines */
            gap: 0.5rem;             /* spacing between items */
            }
            h1 {
            text-wrap: balance;
            }
            .stColumn:has(.downloadButton) {
            flex: 0 1 256px;
            }
            .stColumn:has(.downloadButton) > div {/* The column containing the two buttons */
            flex-direction: column;
            align-items: flex-end;
            font-size: 1.5rem;
            min-width: 100px;
            flex-wrap: nowrap;
            gap: 4px;
            }
            .stColumn:has(.downloadButton) .stElementContainer:has(button) {/* The two buttons */
            min-width: 100px;
            word-break: break-word;
            }
            .stElementContainer:has(.downloadButton) {/* The two empty divs having the class */
            width: 0;
            min-width: 0;
            }
        </style>""")
col_header1, col_header2 = st.columns([1, 1])
col_header1.title("Energibalance-model af Jordens klima")
col_header2.html("<div class='downloadButton'></div>") # style='height: 32px'
if col_header2.button("Lav datafiler til download", type="primary"):
    if "axes_funcs" not in st.session_state or "summaries" not in st.session_state:
        st.error("Kør først simuleringen før du kan lave filer til download.")
    else:
        sim_info = print_simulation_info(st.session_state["config"], st.session_state["params"])
        fig, _, clean_summary = generate_outputs_data(st.session_state["axes_funcs"], st.session_state["summaries"], sim_info=sim_info)
        png_buf = make_png(fig)
        col_header2.download_button("Download figurer (.png)", png_buf, file_name="Klimamodel_figurer.png", on_click="ignore", mime="image/png")
        col_header2.download_button("Download opsummering (.txt)", clean_summary.encode('utf-8'), file_name="Klimamodel_opsummering.txt", on_click="ignore")
col_header2.html("<div class='downloadButton'></div>")
col_header2.download_button("Download modelbeskrivelse (.pdf)", open(os.path.join(os.path.dirname(__file__),"Energybalancemodel.pdf"), "rb").read(), file_name="Energybalancemodel.pdf", on_click="ignore", mime="application/pdf", type="primary")

# --- Toggle for showing advanced parameters ---
st.toggle("Vis alle parametre", key="show_all_params", value=False, on_change=show_params_expander)

st.html("<i class='no-gap'>Klik på knapperne nedenfor for at vælge forudindstillinger for forskellige eksperiment-tilstande.</i>")

# Top buttons in one row
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
with col_btn1:
    # --- Default button ---
    if btns.button("Standardtilstand", icon="🔄"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS)
        set_keyed_inputs(DEFAULT_CONFIG)
with col_btn2:
    # --- Seasonal Variation button ---
    if btns.button("Sæsonvariation", icon="🌱"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=20))  # No forcing in seasonal variation mode
        set_keyed_inputs(DEFAULT_CONFIG | dict(years=50, dt_years=1/24, modes=["SeasonalVariation"]))
with col_btn3:
    # --- Variable Sea Depth button ---
    if btns.button("Variabel havdybde", icon="🌊"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=None))  # Sea depth is irrelevant in this mode
        set_keyed_inputs(DEFAULT_CONFIG | dict(modes=["VariableSeaDepth"]))
with col_btn4:
    # --- Seasonal Variation + Variable Sea Depth button ---
    if btns.button("🌱🌊 Sæsonvariation + Variabel havdybde"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=None)) # No forcing and sea depth is irrelevant
        set_keyed_inputs(DEFAULT_CONFIG | dict(years=50, dt_years=1/24, modes=["SeasonalVariation", "VariableSeaDepth"]))

with st.form("input_form"):
    st.html("<i class='no-gap'>Eller indstil parametre og opsætning manuelt nedenfor.</i>")
    col1, col2 = st.columns(2) # Spacing column in the middle
    with col1:
        with st.expander("🧮 Parametre", expanded=st.session_state.expand):
            st.header("Parametre")
            # Overwrite params dict with user input when form is submitted
            st.number_input(r"F: Ekstra strålingspåvirkning (W/m²)", value=DEFAULT_PARAMS["F"], key="F", step=1.0)
            st.number_input("SD: Varmekapacitet i meter havdybde (m)", value=DEFAULT_PARAMS["SD"], key="SD", step=10)
            st.number_input("D0: Diffusionskoefficient (m²/s)", value=DEFAULT_PARAMS["D0"], key="D0", step=0.1)
            st.number_input("T0: Initial temperatur (K)", value=DEFAULT_PARAMS["T0"], key="T0", step=10)
            if st.session_state.show_all_params:
                st.number_input("S0: Solindstråling under kontrolperiode (W/m²)", value=DEFAULT_PARAMS["S0"], key="S0", step=50)
                st.number_input("S1: Solindstråling efter kontrolperiode (W/m²) (lad stå tom for ingen ændring)", value=DEFAULT_PARAMS["S1"], key="S1", step=50)
                st.number_input("k1: Temperatursensitivitet for isdannelse (K⁻¹)", value=DEFAULT_PARAMS["k1"], key="k1", step=0.01)
                st.number_input("k2: Temperatursensitivitet for diffusivitet (K⁻¹)", value=DEFAULT_PARAMS["k2"], key="k2", step=0.005)
                st.number_input("k3: Feedbackstyrke af drivhuseffekten", value=DEFAULT_PARAMS["k3"], key="k3", step=0.1)
            for key in DEFAULT_PARAMS.keys():
                value = st.session_state.get(key)
                if value is not None: st.session_state.params[key] = value

    with col2:
        with st.expander("⚙️ Opsætning"):
            st.header("Opsætning")
            # Overwrite config dict with user input when form is submitted
            st.number_input("Simuleringstid (år)", value=DEFAULT_CONFIG["years"], key="years", step=50, min_value=1, max_value=1000)
            st.number_input("Kontrolperiode (år) (lad stå tom for halvdelen af simuleringstiden)", value=DEFAULT_CONFIG["ctrl_years"], key="ctrl_years", step=50, min_value=0, max_value=1000)
            st.number_input("Tidsskridt (år)", value=DEFAULT_CONFIG["dt_years"], key="dt_years", step=0.01, min_value=0.01, max_value=10.0)
            st.number_input("Antal gitterpunkter", value=DEFAULT_CONFIG["nx"], key="nx", step=100, min_value=10, max_value=1000)
            st.html("""<style> 
                    /* --- Highlight of selected options in multiselect --- */
                    span[data-baseweb="tag"] {
                        background-color: #80b080 !important;
                    }
                </style>""")
            modes       = st.multiselect("Tilstande", options=["SeasonalVariation", "VariableSeaDepth"], default=DEFAULT_CONFIG["modes"], key="modes")
            for key in DEFAULT_CONFIG.keys():
                value = st.session_state.get(key)
                if value is not None:
                    st.session_state.config[key] = value

    submitted = st.form_submit_button("▶️ Kør simulation med opdaterede parametre og opsætning", width="stretch")

# Run simulation and show outputs
st.session_state.axes_funcs, st.session_state.summaries = run(st.session_state.params, st.session_state.config)
col_out1, col_out2 = st.columns(2, border=True)
with col_out1:
    plot_in_tabs(st.session_state.axes_funcs, st.session_state.summaries)
with col_out2:
    st.subheader("Sammenfatning af experiment")
    if st.session_state.summaries:
        st.html(make_summary(st.session_state.summaries))

    # st.markdown("---")
    # st.header("Om denne app")
    # st.markdown("""
    #     Denne app er lavet af [Johan Skak](https://github.com/JohanSkak) 
    #     i samarbejde med Egil Kaas, professor i klimafysik ved Niels Bohr Institutet, Københavns Universitet,
    #     og Ludvig Pio, studentermedhjælper ved samme institut.
    #     Formålet med appen er at give en simpel introduktion til, hvordan Jordens klima fungerer,
    #     og hvordan forskellige faktorer påvirker klimaet.
    #     Appen er baseret på en simpel energibalance model, som simulerer Jordens klima over tid.
    #     Kildekoden til appen og modellen kan findes på [GitHub](https://github.com/JohanSkak/Klimamodeller).
    #     """)













# import json

# scroll_tabs = components.declare_component(name="scroll_tabs", path="./frontend")

# def tab_component(titles, selected):
#     # Escape data for JS
#     titles_json = json.dumps(titles)
#     selected_json = json.dumps(selected)

#     html = f"""
#         <html>
#         <head>
#         <style>
#             body {{
#                 font-family: sans-serif;
#                 margin: 0;
#             }}
#             #tabs {{
#                 overflow-x: auto;
#                 white-space: nowrap;
#                 padding: 4px;
#                 border-bottom: 1px solid #ccc;
#                 width: 500px;
#             }}
#             .tab {{
#                 display: inline-block;
#                 padding: 6px 12px;
#                 margin: 0 2px;
#                 border-radius: 6px;
#                 cursor: pointer;
#                 background: #eee;
#                 color: #333;
#                 user-select: none;
#                 white-space: nowrap;
#             }}
#             .tab.selected {{
#                 background: #0078ff;
#                 color: white;
#                 font-weight: 600;
#             }}
#         </style>
#         </head>
#         <body>
#         <div id="tabs"></div>
#         <script>
#                 const titles = {titles_json};
#                 const selected = {selected_json};
#                 const container = document.getElementById("tabs");

#                 titles.forEach(t => {{
#                     const el = document.createElement("div");
#                     el.className = "tab" + (t === selected ? " selected" : "");
#                     el.textContent = t;
#                     el.onclick = () => {{
#                         window.parent.postMessage(
#                         {{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: t}},
#                         "*"
#                         );
#                     }};
#                     container.appendChild(el);
#                 }});

#                 // Scroll to the selected one after rendering
#                 const sel = container.querySelector(".selected");
#                 if (sel) sel.scrollIntoView({{ behavior: "smooth", inline: "center" }});
#         </script>
#         </body>
#         </html>
#     """
#     return scroll_tabs(html=html, height=60)

# # Example Streamlit usage
# titles = ["Global Mean Surface Temperature", "Equator", "Denmark (56°N)", "North pole", "South pole"]
# selected = st.session_state.get("choice", titles[0])
# selected

# clicked = tab_component(titles, selected)

# if clicked != selected:
#     var = clicked

# st.write("Selected:", var)