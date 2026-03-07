import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import re, io, os, csv
from main import main
from outputs import generate_outputs_data, print_simulation_info

class ButtonGroup:
    """
    A group of buttons with associated special names and functions.

    Parameters:
        special_names: A list of identifiers for certain buttons.
        special_funcs: A list of functions corresponding to each special identifier. Each function takes a boolean argument indicating whether its button was clicked.

    Application:
        btns = ButtonGroup(special_names=None, special_funcs=None): Creates a Streamlit button group with extra functionality.
        Then use btns.button(label, name=None, **kwargs) to create buttons from the group.
        When the button is clicked, it first runs any on_click function provided in kwargs, then runs all the stored special functions with boolean arguments indicating whether their button was clicked.
        Useful for creating multiple buttons that trigger opposing behaviors in the app. Say, one button to enable a mode and the rest to disable it.
    """
    def __init__(self, special_names=None, special_funcs=None):
        """Store the special_names and special_funcs lists, ensuring they are lists of the same length."""
        self.special_names = special_names if isinstance(special_names, list) else ( [special_names] if special_names else [] )
        self.special_funcs = special_funcs if isinstance(special_funcs, list) else ( [special_funcs] if special_funcs else [] )
        if len(self.special_names) != len(self.special_funcs):
            raise ValueError("special_names and special_funcs must have the same length")

    def button(self, label, name=None, **kwargs):
        """Creates a Streamlit button. When clicked, runs the associated special function (if any) along with any on_click function provided in kwargs."""
        def run_first(inFunc=None):
            """Run the inFunc first (if any), then the special functions with boolean arguments indicating whether their button was clicked."""
            if inFunc: inFunc()
            # Call the functions with True/False depending on whether the special button was clicked
            for _name, func in zip(self.special_names, self.special_funcs):
                func(_name == name)
        
        # Wraps any on_click functions provided in kwargs into run_first
        inFunc = kwargs.pop("on_click", None)
        kwargs["on_click"] = lambda: run_first(inFunc)

        return st.button(label, **kwargs)

@st.cache_data
def run(params, config):
    """
    Run the main simulation with given params and config in app mode.
    The function is cached to avoid re-running the simulation if inputs haven't changed.
    """
    return main(config | {"output_dir": ""}, params, app_mode=True) # output_dir is unused in app mode but must be a string 

def plot_in_tabs(axes_funcs, hash_code):
    """
    Given a list of axes functions, create plots and display them in Streamlit tabs.
    
    Parameters:
        axes_funcs: List of axes-functions that take an axis and either draws on the axis or returns an animation-function.
            An animation function should take size and return HTML for embedding the animation.
        hash_code: A hash code for caching purposes unique for each config-parameter set.

    Method:
        - Create figs and titles using make_plots_and_titles. The figs list can contain either matplotlib figures or animation-functions (for lazy evaluation and caching).
        - If no figs are created (hypothetical), display an info message and return.
        - Create a selectbox for choosing which figure to display based on the titles, remembering the last choice if possible.
    
    Called in col_out1
    """
    figs, titles = make_plots_and_titles(axes_funcs)
    if not figs:
        st.info("Ingen figurer at vise. Prøv at ændre parametre eller opsætning.")
        return
    
    # Remember last choice if possible. If "choice" not in titles, default to first figure. This commonly happens when changing modes.
    index = titles.index(st.session_state["choice"]) if st.session_state.get("choice") in titles else 0
    # When user selects a figure, update the session state "choice" is set to that title
    st.selectbox("Vælg figur:", titles, label_visibility="collapsed", key="choice", index=index)
    # Set the current figure based on the selected title. Here "choice" is guaranteed to be in titles.
    fig = figs[titles.index(st.session_state["choice"])]
    size = 6 # Size in inches for animation
    # If fig is not a figure but an animation-function, call it to get the HTML and embed it
    if not isinstance(fig, plt.Figure):
        html = call_animate_on_earth(fig, size, hash_code) # Call the function to get the animation. This is cached.
        # Embed the HTML with custom styles to ensure proper aspect ratio and spacing
        st.html("<style>.stElementContainer:has(iframe) {aspect-ratio: 1 / 1.25; height: auto; margin-top: -1em;} div[aria-label='Vælg figur:'] {margin: 0;}</style>")
        # Styling to place inside the iFrame created by components
        style = """<style>.animation {width: 100%} .animation img {width: 100%; margin-top: -15px; margin-bottom: -20px}
                    .anim-controls > input {width: 100% !important} body {margin: 0}</style>"""
        components.html(style+html) # Embed the animation HTML
    else: # Otherwise, fig is a matplotlib figure; display it directly
        st.pyplot(figs[titles.index(st.session_state["choice"])])
    plt.close("all") # Close all figures to save memory

@st.cache_data
def call_animate_on_earth(_func, size, hash_code):
    """
    A cached wrapper to call an animation function with given size.
    
    Called by plot_in_tabs
    """
    return _func(size)

def make_plots_and_titles(_axes_funcs):
    """
    Creates a figure for each axes-function and calls them to draw onto the axes and to retrieve their titles.
    If the axes-function does not return anything (i.e. actually draws on the axis) then append the figure, otherwise append the returned animation-function.
    Note: it is time-consuming to create and draw all the plots when only one is needed, but currently this is necessary to get the titles.
    
    Called by plot_in_tabs
    """
    figs = []
    titles = []
    for ax_func in _axes_funcs:
        fig, ax = plt.subplots()
        ani_func = ax_func(ax)
        if ani_func is None:
            figs.append(fig)
        else:
            figs.append(ani_func)
        titles.append(ax.get_title())
    return figs, titles

def make_summary(summaries):
    """
    Make a combined summary in HTML from a list of summaries. Formatting is tailored specially to the summaries of the Output classes.
    
    Called in col_out2
    """
    if not summaries: return ""
    summary = "\n\n".join(summaries)
    style, str = ansi_to_html(summary) # Replaces the ANSI color-codes with CSS styling
    return f"{style}<div id='summary'>{format_terminal_output(str)}</div>"

def ansi_to_html(str):
    """
    Convert basic ANSI colors to HTML spans and style the summary container

    Parameters:
        str: A (summary) string to format
    
    Returns:
        A tuple containing the style string and the formatted HTML string. The output is supposed to be used like this st.html(style+f"<div id='summary'>{str}</div>").

    Called by make_summary
    """
    # Dict of ansi-code to color. The only colors in current use are red and blue which have been custom chosen for nice styling
    colors = {
        "31": "#ff5100", # orange-red
        "32": "green",
        "33": "yellow",
        "34": "#00aeff", # ice-blue
        "35": "magenta",
        "36": "cyan",
        "90": "gray",
    }
    # Styling of the summary container
    style = """
        <style>
        #summary {
                font-size: 0.95rem;
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 0.5rem;
                background: #1ea8493d;  /* Light green background */
                border-radius: 0.5rem;
                overflow-x: auto; /* Enable horizontal scrolling if needed */
            }
        </style>
    """
    # Style elements with the class = ANSI-codenum with their respective color
    color_style = ""
    for code, color in colors.items():
        color_style += f"._{code} {{color: {color};}}\n"
    style += f"<style>\n{color_style}</style>"

    # Replace ANSI-codes - \033[<n>m - with <span class="_<n>"> which get their style from color_style
    for code, color in colors.items():
        str = re.sub(fr"\033\[{code}m", f"<span class='_{code}'>", str) #Uses regex to find and replace
    # Reset code / end span-tag
    str = re.sub(r"\033\[0m", "</span>", str)
    # Warn if any unhandled codes remain
    if re.search(r"\033\[\d+m", str):
        st.info("Warning: Unhandled ANSI codes remain in summary.")
    return style, str

def format_terminal_output(text):
    """
    Format terminal output text into HTML with blocks of aligned columns for key-value pairs. Encapsulates decimal numbers in boxes with fixed width in order to align the text among multiple lines.

    Parameters:
        text: A string containing the typical terminal output
    
    Returns:
        A string where the text has been broken into blocks each block containing exactly one colon per line. Each block is grid-formatted with two columns: before and after the colon

    Called by make_summary
    """
    # Split the text into lines and remove any white space at the end of the lines
    lines = [l.rstrip() for l in text.splitlines()]
    html_blocks = [] # List of HTML-formatted blocks
    block = [] # A placeholder for a block i.e. list of lines

    def make_space(string, pattern, space):
        """Wrap all occurrences of pattern in string with a span that reserves space."""
        return re.sub(fr"({pattern})", fr"<span style='display:inline-block; min-width:{space}em; text-align:right;'>\1</span>", string)

    def estimate_pixel_width(s: str, px_per_char=8, px_per_wide=12, px_per_medium_wide=10, px_per_narrow=5):
        """Estimate pixel width of string s based on character types."""
        wide = sum(1 for c in s if c in "W@")
        medium_wide = sum(1 for c in s if c in "mM")
        narrow = sum(1 for c in s if c in "iIl.,;:'|!1 ()-°")
        normal = len(s) - wide - narrow - medium_wide
        return wide * px_per_wide + medium_wide * px_per_medium_wide + narrow * px_per_narrow + normal * px_per_char + 5

    def flush_block(block):
        """Render one logical block"""
        if not block:
            return "" # Empty block, i.e. consecutive empty lines
        # Check if all lines have exactly one colon
        if all(line.count(":") == 1 for line in block):
            # Compute (approximately) maximal width of each column over all lines in pixels
            max_key_px = max(estimate_pixel_width(line.split(":", 1)[0]) for line in block)
            max_val_px = max(estimate_pixel_width(re.sub(r'<[^>]+>', '', line.split(":", 1)[1])) for line in block) #re.sub(r'<[^>]+>',... removes HTML tags for width estimation
            # Create grid block with two columns. Wrapping is enabled with maximally two lines. This layout is quite finicky but works reasonably well.
            html = f"""<div class='block' style='grid-template-columns: minmax({int(max_key_px*0.6)}px, max-content) minmax({int(max_val_px*0.5)}px, max-content);'>"""
            # For each line, split into key and value at the colon and further make spaces for numbers
            for line in block:
                key, val = line.split(":", 1)
                # Highlight numbers with inline-block spans (optional). Regex explanation:
                # \s*   Optional leading whitespace; [+-]? Optional sign; \d+   One or more digits (the integer part);
                # [.,]  Decimal separator: either "." or ","; \d{2} Exactly two decimal digits; (?!(\d|[a-z])) Negative lookahead: next char must NOT be a digit or a letter
                val_html = make_space(val, r"\s*[+-]?\d+[.,]\d{2}(?!(\d|[a-z]))", 2.9) # Numbers like 123.45 or -0.52
                val_html = make_space(val_html, r"\s*[+-]?\d+[.,]\d{1}(?!(\d|[a-z]))", 2.3) # Numbers like 123.4 or -0.5
                val_html = make_space(val_html, r"[A-Z][a-z]{2}", 1.8) # 3-letter month abbreviations like Jan, Feb, Mar
                html += f"""
                <div class='key'>{key.strip()}<b>:</b></div>
                <div class='value'>{val_html}</div>
                """
            html += "</div><br>"
            return html
        else:
            # Otherwise, flat text block
            return "<div style='text-wrap: nowrap;'>" + "<br>".join(block) + "</div><br>"

    # Loop through all lines and append to block. When an empty line appears call flush_block and reset block.
    for line in lines + [""]:
        if not line.strip(): # Empty line indicates new block
            html_blocks.append(flush_block(block))
            block = []
        else:
            block.append(line)
    html_blocks[-1] = html_blocks[-1].rstrip("<br>") # Remove very last <br>
    
    block_style = """<style>
        .block {
            display:grid;       /* Grid styling for creating two columns with controlled width. The width is set inside flush_block */
            line-height: 1.2;
        }
        .block .key { 
            grid-column: 1;     /* The keys are in the first column */
            text-align: left;
            text-wrap: balance; /* Break lines in a balanced way (almost equal line width) */
            align-self: start;  /* Align to top */
        }
        .block .value {
            grid-column: 2;     /* The values are in the second column */
            text-align: right;
            align-self: end;    /* Align to bottom */
        }
        .block > div {          /* Both the key and value containers */
            margin-bottom: 0.6em;
            white-space: normal;
        }
        </style>"""
    # Combine all
    return block_style + ''.join(html_blocks)

def set_keyed_inputs(dict):
    """Saves all items in dict into st.session_state.
    Used specifically for params and config dicts to set the input widgets when a preset button is clicked."""
    for k, v in dict.items():
        st.session_state[k] = v # Update forms and create keys if they don't exist yet (toggle off)

def make_png(fig):
    """Save a matplotlib figure to a PNG in a BytesIO buffer and return the buffer (basically a file-like object)."""
    buf = io.BytesIO() # Create a bytes buffer
    fig.savefig(buf, format="png", bbox_inches='tight') # Save figure to buffer
    buf.seek(0) # Rewind buffer to the beginning
    return buf

def upload_file_menu():
    """
    Create a file upload menu for uploading custom forcing data in CSV format. The uploaded data is validated and stored in session state before being passed to the simulation config.

    Method:
    - Define a helper function use_default_forcing_data to reset to default forcing data.
    - Define a helper function check_file_format to validate the uploaded data format.
    - Initialize session state variables for storing the forcing data and filename.
    - Create a toggle switch to enable/disable custom forcing data upload.
    - If the toggle is enabled, display a file uploader widget for CSV files. Else, reset to default forcing data.
    - If a file is uploaded, read it as CSV and validate the uploaded file size and format using check_file_format. Else, reset to default forcing data.
    - If valid, store the data and filename in session state; otherwise, display an error message.

    Called in the main app just before running the simulation if VariableForcing mode is selected.
    """
    def use_default_forcing_data():
        """Reset to default forcing data by clearing session state."""
        st.session_state.forcing_data = None
        st.session_state.forcing_filename = None
    
    def check_file_format(data):
        """
        Check that the uploaded file has the correct format: two columns, numeric data.

        Parameters:
            data: List of rows, where each row is a list of strings (from CSV reader)
        
        Returns:
            bool: True if the format is valid, False otherwise.
        """
        try:
            for row in data:
                if len(row) != 2: # Must have exactly two columns
                    return False
                float(row[0])  # Check if first column is numeric
                float(row[1])  # Check if second column is numeric
            return True
        except ValueError: # Conversion to float failed indicating non-numeric data
            return False

    # Initialize session storage: ensure keys exist
    if "forcing_data" not in st.session_state:
        st.session_state.forcing_data = None
    if "forcing_filename" not in st.session_state:
        st.session_state.forcing_filename = None

    # Toggle switch for the user to enable/disable custom forcing data upload
    use_custom_forcing = st.toggle("Brug egen forceringsdata", value=False)

    # Reset and quit if toggled off
    if not use_custom_forcing:
        use_default_forcing_data()
        return
    
    # Create file uploader widget
    uploaded_file = st.file_uploader(
        "Upload din egen CSV-fil med forceringsdata",
        help = "Filen skal være en CSV med to kolonner: år og forcering i W/m². Første række kan være en header.",
        type=["csv"], 
        key="forcing_uploader"
    )

    # Reset and quit if no file is uploaded
    if uploaded_file is None:
        use_default_forcing_data()
        return
    
    # Simple malware prevention: limit size and enforce CSV reading
    # The config.toml file in .streamlit folder also sets max upload size to 1 MB (which is the minimum limit allowed by Streamlit)
    MAX_SIZE_KB = 100
    if uploaded_file.size > MAX_SIZE_KB * 1024:
        st.error(f"Filen er for stor (> {MAX_SIZE_KB} KB).")
        return

    try:
        # Read as CSV safely
        reader = csv.reader(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        # Skip header if non-numeric
        first_row = next(reader)
        if check_file_format([first_row]): st.session_state.forcing_data = [first_row] + [row for row in reader]
        else: st.session_state.forcing_data = [row for row in reader]
        st.session_state.forcing_filename = uploaded_file.name
        # Validate format
        if not check_file_format(st.session_state.forcing_data):
            use_default_forcing_data()
            st.error("Ugyldigt filformat. Sørg for at filen har to kolonner med numeriske data.")
    except Exception as e:
        use_default_forcing_data()
        st.error(f"Kunne ikke læse CSV: {e}")

# Function to set run_away
def set_run_away(Bool):
    st.session_state.run_away = Bool
# Create ButtonGroup instance where most buttons set run_away to False except the runaway button
btns = ButtonGroup("run_away", set_run_away)


# ---- Begin Streamlit app ----
DEFAULT_PARAMS = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288, SD=250, S0=1365, S1=None, F=4.0)
DEFAULT_CONFIG = dict(years=1000, ctrl_years=None, dt_years=1.0, nx=200, modes=[], forcing_file='ForcingHistory.csv')

# Initialize with default values
if "params" not in st.session_state:
    st.session_state.params = DEFAULT_PARAMS.copy()
if "config" not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()

# Configer page layout
st.set_page_config(page_title="Energibalancemodel af Jordens klima", page_icon="🌍",)
st.html("""
    <style>            
        /* Change the width and max width of the main content area */
        .block-container {
            max-width: 90%;
            width: 1500px;
        }
        .stElementContainer:has(.no-gap) {  /* Remove top and bottom margin for elements with class no-gap */
            margin-top: -1em;
            margin-bottom: -1em;
        }
    </style>
""")

# Sidebar for preset experiments
with st.sidebar:
    st.markdown("---") # Horizontal line
    st.html("<style> h2 {padding: 0 !important;} </style>") # No padding in title
    st.header("Forudindstillinger")
    st.html("<i>Nogle forudindstillinger for simple eksperimenter.</i>")
    if btns.button("Standardtilstand", icon="🔄"):
        set_keyed_inputs(DEFAULT_PARAMS)
        set_keyed_inputs(DEFAULT_CONFIG)
    if btns.button("Nul-diffusion", icon=":material/mode_fan_off:"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(D0 = 0.0, F=0.0))  # No forcing in seasonal variation mode
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile" # Set default figure to show
    if btns.button("Havdybde 2000 m", icon="🌊"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=2000))
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Global Mean Surface Temperature" # Set default figure to show
    if btns.button("Havdybde 20 m", icon="🪨"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=20))
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Global Mean Surface Temperature" # Set default figure to show
    if btns.button("Snebold-Jorden", icon="❄️"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, T0=245))  # No forcing and no sea depth
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile" # Set default figure to show
    if btns.button("Løbsk drivhuseffekt", name="run_away", icon="🔥"):
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=20, k3=1.1))  # Strong forcing and strong feedback
        set_keyed_inputs(DEFAULT_CONFIG)
        st.session_state.choice = "Temperature Profile" # Set default figure to show
    if st.session_state.get("run_away", False): # Show input field if the last button click was runaway
        st.html("<style> div.stHorizontalBlock:has(.slim-container) {gap: 0.3em} </style>") # Reduce gap
        col_run_away1, col_run_away2 = st.columns([1, 2])
        col_run_away1.html("<div class='slim-container'>Indstil k3:</div>")
        col_run_away2.number_input("Label", label_visibility="collapsed", value=1.1, key="k3_runaway", step=0.1)
        st.session_state["k3"] = st.session_state["k3_runaway"]


# --- Header and download buttons ---
st.html("""<style>
            .stHorizontalBlock:has(.downloadButton) {/* The header with title and download buttons */
            display: flex;
            flex-wrap: wrap;            /* allow multiple lines */
            justify-content: space-between;
            align-items: flex-end;      /* align bottoms of wrapped lines */
            gap: 0.5rem;                /* spacing between items */
            }
            h1 {
            text-wrap: balance;
            }
            .stColumn:has(.downloadButton) {
            flex: 0 1 256px;            /* Allow column with download buttons to shrink but not grow too large */
            }
            .stColumn:has(.downloadButton) > div {/* The column containing the two buttons */
            flex-direction: column;     /* Stack buttons vertically */
            align-items: flex-end;      /* Align buttons to the right */
            font-size: 1.5rem;
            min-width: 100px;
            gap: 4px;
            }
            .stElementContainer:has(.downloadButton) {/* The two empty divs having the class */
            width: 0;
            min-width: 0;
            }
        </style>""")
col_header1, col_header2 = st.columns([1, 1]) # Two columns: one for title, one for download buttons
col_header1.title("Energibalance-model af Jordens klima")
col_header2.html("<div class='downloadButton'></div>") # Empty container solely for styling purposes (via its class tag)
if col_header2.button("Lav datafiler til download", type="primary"):
    sim_info = print_simulation_info(st.session_state["config"], st.session_state["params"], app_mode=True)
    # Remove the animation from the list of axes functions
    axis_funcs = [ax_func for ax_func in st.session_state["axes_funcs"] if not ax_func.__name__ == "panel_wrapper"]
    fig, _, clean_summary = generate_outputs_data(axis_funcs, st.session_state["summaries"], sim_info=sim_info)
    png_buf = make_png(fig)
    col_header2.download_button("Download figurer (.png)", png_buf, file_name="Klimamodel_figurer.png", on_click="ignore", mime="image/png")
    col_header2.download_button("Download opsummering (.txt)", clean_summary.encode('utf-8'), file_name="Klimamodel_opsummering.txt", on_click="ignore")
col_header2.html("<div class='downloadButton'></div>") # Another empty container for spacing
col_header2.download_button("Download modelbeskrivelse (.pdf)", open(os.path.join(os.path.dirname(__file__),"Energybalancemodel.pdf"), "rb").read(), file_name="Energybalancemodel.pdf", on_click="ignore", mime="application/pdf", type="primary")

# --- Toggle for showing advanced parameters ---
st.toggle("Vis alle parametre", key="show_all_params", value=False)

st.html("<i class='no-gap'>Klik på knapperne nedenfor for at vælge forudindstillinger for forskellige simuleringstilstande.</i>")

# --- Preset buttons ---
# CSS to balance text wrapping in button paragraphs
st.html("<style> button p {text-wrap: balance;}</style>")
col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)
with col_btn1:
    # --- Default button ---
    if btns.button("Standardtilstand", icon="🔄"):
        # Set the relevant keyed input widgets to their default values
        set_keyed_inputs(DEFAULT_PARAMS)
        set_keyed_inputs(DEFAULT_CONFIG)
with col_btn2:
    # --- Seasonal Variation button ---
    if btns.button("Sæsonvariation", icon="🌱"):
        # Set the keyed input widgets to mode-specific default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=20))  # No forcing in seasonal variation mode
        set_keyed_inputs(DEFAULT_CONFIG | dict(years=50, dt_years=1/24, modes=["SeasonalVariation"]))
with col_btn3:
    # --- Variable Sea Depth button ---
    if btns.button("Variabel havdybde", icon="🌊"):
        # Set the keyed input widgets to mode-specific default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(SD=0))  # Sea depth is irrelevant in this mode
        set_keyed_inputs(DEFAULT_CONFIG | dict(modes=["VariableSeaDepth"]))
with col_btn4:
    # --- Seasonal Variation + Variable Sea Depth button ---
    if btns.button("🌱🌊 Sæsonvariation + Variabel havdybde"):
        # Set the keyed input widgets to mode-specific default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0.0, SD=0)) # No forcing and sea depth is irrelevant
        set_keyed_inputs(DEFAULT_CONFIG | dict(years=50, dt_years=1/24, modes=["SeasonalVariation", "VariableSeaDepth"]))
with col_btn5:
    # --- Variable Forcing button ---
    if btns.button("Forceringsdata", icon="📈"):
        # Set the keyed input widgets to mode-specific default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0, SD=20))  # Sea depth value to fit ERA5 data
        set_keyed_inputs(DEFAULT_CONFIG | dict(ctrl_years=100, modes=["VariableForcing"]))
        st.session_state.choice = "Global Mean Surface Temperature and Total Radiative Forcing" # Set default figure to show for this mode
with col_btn6:
    # --- Variable Forcing button ---
    if btns.button("📈🌡️ Forceringsdata + Temperaturdata"):
        # Set the keyed input widgets to mode-specific default values
        set_keyed_inputs(DEFAULT_PARAMS | dict(F=0, SD=20))  # Sea depth value to fit ERA5 data
        set_keyed_inputs(DEFAULT_CONFIG | dict(ctrl_years=100, modes=["VariableForcing", "HistoricalData"]))
        st.session_state.choice = "Global Mean Surface Temperature and Observed Temperature Anomaly" # Set default figure to show for this mode

# --- Input form for parameters and config ---
with st.form("input_form"): # Note: A form does not auto-submit when inputs change
    st.html("<i class='no-gap'>Eller indstil parametre og opsætning manuelt nedenfor.</i>")
    col1, col2 = st.columns(2)
    # Parameter inputs
    with col1:
        with st.expander("🧮 Parametre", expanded=st.session_state.get("show_all_params", False)):
            # Overwrite params dict with user input when form is submitted
            st.header("Parametre")
            if not "VariableForcing" in st.session_state.get("modes", []):
                st.number_input(r"F: Ekstra strålingspåvirkning (W/m²)", value=DEFAULT_PARAMS["F"], key="F", step=1.0)
            if not "VariableSeaDepth" in st.session_state.get("modes", []):
                st.number_input("SD: Varmekapacitet i meter havdybde (m)", value=DEFAULT_PARAMS["SD"], key="SD", step=10, min_value=1)
            st.number_input("D0: Diffusionskoefficient (m²/s)", value=DEFAULT_PARAMS["D0"], key="D0", step=0.1)
            st.number_input("T0: Initial temperatur (K)", value=DEFAULT_PARAMS["T0"], key="T0", step=10, min_value=0)
            if st.session_state.show_all_params: # Show more parameters
                st.number_input("S0: Solindstråling under kontrolperiode (W/m²)", value=DEFAULT_PARAMS["S0"], key="S0", step=50)
                st.number_input("S1: Solindstråling efter kontrolperiode (W/m²) (lad stå tom for ingen ændring)", value=DEFAULT_PARAMS["S1"], key="S1", step=50)
                st.number_input("k1: Temperatursensitivitet for isdannelse (K⁻¹)", value=DEFAULT_PARAMS["k1"], key="k1", step=0.01)
                st.number_input("k2: Temperatursensitivitet for diffusivitet (K⁻¹)", value=DEFAULT_PARAMS["k2"], key="k2", step=0.005)
                st.number_input("k3: Feedbackstyrke af drivhuseffekten", value=DEFAULT_PARAMS["k3"], key="k3", step=0.1)
            for key in DEFAULT_PARAMS.keys(): # Update params dict in session state with non-None input values
                value = st.session_state.get(key) # Retrieve the input value; returns None if the key does not exist yet
                if value is not None or key in ["S1"]:  # Allow S1 to be None (indicating no change in solar radiation after control period)
                    st.session_state.params[key] = value
    # Config inputs
    with col2:
        with st.expander("⚙️ Opsætning"):
            # Overwrite config dict with user input when form is submitted
            st.header("Opsætning")
            if not "VariableForcing" in st.session_state.get("modes", []):
                st.number_input("Simuleringstid (år)", value=DEFAULT_CONFIG["years"], key="years", step=50, min_value=1, max_value=1000)
            # Ensure ctrl_years is at most years and set to default if not set yet.
            if "ctrl_years" not in st.session_state:
                st.session_state.ctrl_years = DEFAULT_CONFIG["ctrl_years"]
            max_ctrl = min(1000, st.session_state.get("years", 1000))
            if st.session_state.ctrl_years is not None and st.session_state.ctrl_years > max_ctrl:
                st.session_state.ctrl_years = max_ctrl
            st.number_input("Kontrolperiode (år) (lad stå tom for halvdelen af simuleringstiden)", value=st.session_state.ctrl_years, key="ctrl_years", step=50, min_value=0, max_value=max_ctrl)
            st.number_input("Tidsskridt (år)", value=DEFAULT_CONFIG["dt_years"], key="dt_years", step=0.01, min_value=0.01, max_value=10.0)
            st.number_input("Antal gitterpunkter", value=DEFAULT_CONFIG["nx"], key="nx", step=100, min_value=10, max_value=1000)
            st.html("""<style> 
                    /* --- Highlight of selected options in multiselect --- */
                    span[data-baseweb="tag"] {
                        background-color: #80b080 !important;
                    }
                </style>""")
            modes = st.multiselect("Tilstande", options=["SeasonalVariation", "VariableSeaDepth", "VariableForcing", "HistoricalData"], default=DEFAULT_CONFIG["modes"], key="modes")
            for key in DEFAULT_CONFIG.keys(): # Update config dict in session state with non-None input values
                value = st.session_state.get(key)
                if value is not None or key in ["ctrl_years"]:  # Allow ctrl_years to be None
                    st.session_state.config[key] = value

    submitted = st.form_submit_button("▶️ Kør simulation med opdaterede parametre og opsætning", width="stretch")

# --- Validate modes compatibility ---
if "VariableForcing" in modes and "SeasonalVariation" in modes:
    st.error("'VariableForcing' og 'SeasonalVariation' kan ikke bruges sammen. Vælg kun én af dem.")
    st.stop()

if "HistoricalData" in modes and "SeasonalVariation" in modes:
    st.error("'HistoricalData' og 'SeasonalVariation' kan ikke bruges sammen. Vælg kun én af dem.")
    st.stop()

# --- File upload for custom forcing data if VariableForcing mode is selected ---
if "VariableForcing" in st.session_state.get("modes", []):
    upload_file_menu() # Create the upload menu
    # If custom forcing data is uploaded, pass it to the config
    st.session_state.config["forcing_data"] = st.session_state.forcing_data

# Run simulation and show outputs
st.session_state.axes_funcs, st.session_state.summaries = run(st.session_state.params, st.session_state.config)
col_out1, col_out2 = st.columns(2, border=True)
with col_out1: # Plotting area
    plot_in_tabs(st.session_state.axes_funcs, (st.session_state.params, st.session_state.config))
with col_out2: # Summary area
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