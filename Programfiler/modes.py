# modes.py
import numpy as np
import outputs
import physics as phys

class Mode:
    def __str__(self):
        return self.__class__.__name__
    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    def check_compatibility(self, modes): pass
    outputs = [] # Is a list of output class instances

class SeasonalVariation(Mode):
    def initialize(self, model):
        years = model.config["years"]
        dt_years = model.config["dt_years"]
        # Override config if necessary
        if dt_years > 1/12: # is a problem with default settings
            warn(f"Time step - \033[4m{dt_years} > 1/12 years\033[0m - is to large to capture seasonal variation. Has been set to half a month.")
            model.config["dt_years"] = 1/24 # half monthly steps
        if years > 1000: # too long computation time # not relevant with default settings
            warn(f"Simulation time - \033[4m{years} > 1000 years\033[0m - is to large for reasonable run time. Has been set to fifty years.")
            model.config["years"] = 50
        model.config["years"] = int(years) if years >= 1 else 1 #Run a whole number of years; at least 1
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4 #Should be 1 / num where num is divisible by 4
        model.config["output_dir"] += "_SeVa" #Modify output directory name

        # Replace insolation kernel
        # See Wikipedia Solar Irradiance
        def seasonal_Q(x, S, model, i):
            t = (i+1) * model.config["dt_years"]  # time in years
            eps = np.deg2rad(23.44) # obliquity
            theta = 2 * np.pi * t   # annual angle
            delta = np.arcsin(np.sin(eps) * np.sin(theta)) # Current declination δ = sin⁻¹(sin ε sin θ)

            # Hour angle at sunrise/sunset
            h0 = np.arccos(np.clip(- x / np.sqrt(1 - x**2 + 1e-10) * np.tan(delta), -1, 1))
            SIr = (S / np.pi) * (h0 * x * np.sin(delta) + np.sqrt(1 - x**2) * np.cos(delta) * np.sin(h0))
            return SIr

        model.funcs['Q_x'] = seasonal_Q # Replaces Q_x with seasonal_Q in ClimateModel object
    
    outputs = [outputs.SeasonalOutput(), outputs.TimeSeriesOutput()]

class VariableSeaDepth(Mode):
    def initialize(self, model):
        model.config["output_dir"] += "_SeaDep" #Modify output directory name

    def step(self, model, i):
        model.C = self.heat_capacity_profile(model.config["nx"], model.T, model.params["k1"])

    def heat_capacity_profile(self, nx, T, k1):
        """
        Compute latitude-dependent effective heat capacities for a sin(lat)-spaced EBM grid.
        
        Parameters
        ----------
        nx : int
            Number of gridpoints (sin(lat) spaced from -1 to 1).
        T : array_like, shape (nx,)
            Surface temperature [K] at each gridpoint (used to estimate ice fraction).
        
        Returns
        -------
        C : ndarray, shape (nx,)
            Heat capacity per unit area [J m^-2 K^-1] at each gridpoint.
        
        Notes
        -----
        - Uses an asymmetric zonal ocean fraction f_ocean(lat) (Southern Hemisphere more ocean).
        - Uses a piecewise-interpolated ocean mixed-layer depth h_ocean(lat) (seasonal-scale).
        - Land is treated as a shallow water equivalent of h_land (default 8 m).
        - Ice fraction is a smooth function of T: full ice when T <= 271 K, none when T >= 275 K.
        
        Physical constants:
        rho = 1000 kg/m^3, cp = 4186 J/kg/K (water)
        
        Documentation / rationale:
        - Mixed-layer climatologies show shallow MLD in tropics (~tens m), shallow subtropical
            stratified minima, deeper seasonal mixing in mid-latitudes (~100-300 m), and large
            seasonal deepening at high latitudes (Monterey & Levitus 1997; de Boyer Montégut 2004).
            See references below. :contentReference[oaicite:0]{index=0}
        - Southern Hemisphere has greater ocean fraction than Northern Hemisphere; we reflect
            that asymmetry in f_ocean(lat). :contentReference[oaicite:1]{index=1}
        """
        T = np.asarray(T)
        assert T.shape == (nx,), "T must be length nx"
        
        # physical constants
        rho = 1000.0        # kg/m^3
        cp = 4186.0         # J/kg/K
        h_land = 8.0        # m, land equivalent (water-equivalent), ~1/30 of deep-ocean reference
        
        # latitude grid in sin(lat) space
        x = np.linspace(-1.0, 1.0, nx)        # sin(latitude)
        lat_rad = np.arcsin(x)                # radians
        lat_deg = np.degrees(lat_rad)         # degrees, negative = southern hemisphere
        
        # --------------------------
        # Ocean mixed-layer depth profile (seasonal-scale, more nuance)
        # knots (latitudes in degrees) and representative depths (m)
        # Explanation:
        #  - tropical/ITCZ: shallow seasonal MLD (20-70 m)
        #  - subtropical stratified belts: local shallow minimum (20-40 m)
        #  - mid-latitudes: more storm-driven deepening (100-250 m)
        #  - high-latitude (poleward of ~70): variable, seasonal deepening possible (50-200 m)
        # These numbers are chosen to represent the *seasonal* MLD climatology (not deep ocean).
        lat_knots = np.array([-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90])
        # more oceanic south: allow slightly deeper high-latitude southern mixing (southern storms)
        # depths [m]
        h_knots = np.array([100.0,   # near South Pole: if ice-free, deep seasonal mixing possible; otherwise will be masked by ice
                            180.0,   # 70S - Southern high lat deep mixing in winter (Southern Ocean)
                            220.0,   # 50S - stormier, deeper seasonal ML
                            140.0,   # 30S - mid to subtropics
                            60.0,    # 15S - subtropical shoal
                            50.0,    # 0   - tropics (warm shallow seasonal ML)
                            60.0,    # 15N - slightly larger than equator
                            40.0,    # 30N - subtropical stratified minimum
                            140.0,   # 50N - northern mid-latitude seasonal deepening (but less than SH)
                            80.0,   # 70N - Arctic ocean (shallower than SH)
                            30.0])   # near North Pole (shallow seasonal ML do to fresh water)
        
        # interpolate mixed-layer depth onto grid
        h_ocean = np.interp(lat_deg, lat_knots, h_knots)
        
        # --------------------------
        # Asymmetric zonal ocean fraction f_ocean(lat)
        # Hard-coded (smooth) zonal ocean fraction knots. Values based on general land/ocean
        # geography: Southern hemisphere has more ocean (esp. 30S-60S), Northern hemisphere has
        # more land at mid-latitudes (Eurasia, North America). These are smooth, empirical values.
        # Source: qualitative/quantitative zonal land fraction diagrams (e.g. land-fraction vs latitude).
        lat_knots_f = np.array([-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90])
        # zonal ocean fraction at knots (0..1). Southern hemisphere has systematically larger ocean fraction.
        f_ocean_knots = np.array([0.0,  # Continent/ice cap
                                0.2,   # 70S -> Mostly Antarctic land, but surrounding Southern Ocean begins
                                0.9,   # 50S -> Southern Ocean dominates
                                0.85,  # 30S -> Still ocean-dominated, only S. America, Africa, Australia
                                0.8,   # 15S -> Mostly ocean
                                0.75,  # 0   -> Continents cut across (Africa, S. America, Indonesia)
                                0.7,   # 15N -> Africa + Asia reduce ocean fraction
                                0.6,   # 30N -> Subtropics: Africa, Asia, N. America
                                0.55,  # 50N -> Eurasia + N. America dominate, but N. Atlantic/Pacific present
                                0.65,  # 70N -> Arctic Ocean exists, though partly enclosed
                                1.0])  # Central Arctic Ocean basin
        f_ocean = np.clip(np.interp(lat_deg, lat_knots_f, f_ocean_knots), 0.0, 1.0)
        
        # --------------------------
        # --- Ice fraction (same formula as in albedo) ---
        # ice_fraction in [0,1]: 1 => full ice cover (ocean behaves like land shallow)
        ice_fraction = np.clip(k1 * (273.15 - T), 0.0, 1.0)
        # effective ocean fraction after accounting for seasonal/persistent ice
        f_ocean_eff = f_ocean * (1.0 - ice_fraction)
        f_land_eff  = 1.0 - f_ocean_eff #Ice like land is assumed to act like h_land water depth
        
        # --------------------------
        # Effective depth: ocean fraction uses h_ocean, land fraction uses h_land
        h_eff = f_ocean_eff * h_ocean + f_land_eff * h_land
        
        # --------------------------
        # Heat capacity per unit area
        C = rho * cp * h_eff   # J m^-2 K^-1
        
        return C

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")
