import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

def plot_on_earth(inLat=None,T=None, ax=None, title="Temperature map of the Earth", cbar_label="Temperature [°C]"):
    """
    Plot a simple temperature map of the Earth as seen from space,
    centered on Africa, with coastlines.
    """
    # --- parameters ---
    res = 400  # resolution (pixels per axis)
    R = 1.0    # Earth radius in arbitrary units

    # --- create normalized grid ---
    x = np.linspace(-R, R, res)
    y = np.linspace(-R, R, res)
    X, Y = np.meshgrid(x, y)

    # mask outside the disk
    mask = X**2 + Y**2 <= R**2

    # --- map (x,y) to spherical coords ---
    # observer above (lat=0, lon=0)
    lat = np.degrees(np.arcsin(Y / R)) * np.sign(Y)       # latitude
    # lon = np.degrees(np.arcsin(X / np.sqrt(R**2-Y**2)))                  # longitude

    # --- temperature field (depends only on latitude) ---
    if inLat is None or T is None:
        T = 5 + 25 * (np.cos(2*np.radians(lat)))          # arbitrary model
    else:
        print(len(lat), len(inLat), len(T))
        T = np.interp(lat, inLat, T)                          # interpolate given profile
    T[~mask] = np.nan                                  # mask outside Earth

    # --- plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(T, extent=(-R, R, -R, R), origin='lower',
                cmap='coolwarm', interpolation='bilinear')

    # draw circular outline
    circle = plt.Circle((0, 0), R, color='k', lw=1.2, fill=False)
    ax.add_artist(circle)

    # --- add coastlines (using cartopy's Natural Earth data) ---
    # We can extract coastlines manually as (lon,lat) and project
    reader = shpreader.natural_earth(resolution='110m',
                                    category='physical',
                                    name='coastline')
    for i, record in enumerate(shpreader.Reader(reader).records()):
        coords = np.array(record.geometry.coords)
        # project onto the visible hemisphere (simple orthographic)
        lon_c, lat_c = np.radians(coords[:,0]), np.radians(coords[:,1])
        behind = np.cos(lon_c) < 0
        lon_c[behind] = np.nan
        y_c = np.sin(lat_c)
        x_c = np.cos(lat_c) * np.sin(lon_c)
        for x_c, y_c in split_on_nan(x_c, y_c):
            ax.plot(x_c, y_c, color='black', lw=0.5)

    # --- finalize look ---
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label)
    plt.title(title)
    plt.show()

def animate_on_earth(inLat, T_series, dt_years, interval=80, ax=None, title="Surface Temperature on Earth", cbar_label="°C"):
    """
    Create an animation of temperature maps of the Earth as seen from space,
    centered on 0°N, 0°E, with coastlines.

    Parameters
    ----------
    inLat : array-like
        Latitudes corresponding to the temperature data.
    T_series : array-like, shape (n_times, n_latitudes)
        Time series of temperature profiles at different latitudes.
    interval : int, optional
        Delay between frames in milliseconds. Default is 200.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot. Default is "Surface Temperature on Earth".
    cbar_label : str, optional
        Label for the colorbar. Default is "°C".

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    """
    ax.set_title(title) # Set initial title. Useful in app

    def wrapper(size):
        # --- parameters ---
        res = 400 # resolution (pixels per axis)

        # --- create normalized grid ---
        x = np.linspace(-1, 1, res) # along longitude
        y = np.linspace(-1, 1, res) # along latitude
        X, Y = np.meshgrid(x, y)

        # --- map (x,y) to spherical coords ---
        # observer above equator
        lat = np.degrees(np.arcsin(Y)) # latitude

        # --- plot setup ---
        _, ax = plt.subplots(figsize=(size, size))

        im = draw_earth(ax, np.min(T_series), np.max(T_series), res)

        def update(frame):
            T_t = T_series[frame]
            T_plot = np.interp(lat, inLat, T_t)
            mask = X**2 + Y**2 <= 1
            T_plot[~mask] = np.nan
            im.set_array(T_plot)
            time = f"{frame*dt_years:.0f} years" if dt_years.is_integer() else f"{frame*dt_years:.2f} years"
            ax.set_title(f"{title} ({time})", fontdict={'fontsize': 16*size/6}, pad=20)
            return im,

        ani = FuncAnimation(plt.gcf(), update, frames=np.linspace(0, len(T_series)-1, 51, dtype=int), interval=interval, blit=True)
        
        # --- finalize look ---
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label)
        return ani.to_jshtml()
    
    return wrapper # Lazy evaluation to avoid creating figure when not needed

def draw_earth(ax, min, max, res):
    # draw circular outline
    circle = plt.Circle((0, 0), 1, color='k', lw=1.2, fill=False)
    ax.add_artist(circle)

    # add coastlines
    reader = shpreader.natural_earth(resolution='110m', category='physical', name='coastline')
    for record in shpreader.Reader(reader).records():
        coords = np.array(record.geometry.coords)
        lon_c, lat_c = np.radians(coords[:,0]), np.radians(coords[:,1])
        behind = np.cos(lon_c) < 0
        lon_c[behind] = np.nan
        y_c = np.sin(lat_c)
        x_c = np.cos(lat_c) * np.sin(lon_c)
        for x_c, y_c in split_on_nan(x_c, y_c):
            ax.plot(x_c, y_c, color='black', lw=0.5)

    colors = [[0, 0.7, 1, 1], [1, 1, 1, 1], [1, 0.2, 0, 1]] # blue to white to red
    zero_point = np.clip(-min / (max - min), 0, 1)
    cmap = ListedColormap(np.vstack((np.linspace(colors[0], colors[1], int(256*zero_point)),
                                    np.linspace(colors[1], colors[2], 256 - int(256*zero_point)))))
    return ax.imshow(np.zeros((res,res)), extent=(-1, 1, -1, 1), origin='lower',
                cmap=cmap, interpolation='bilinear', vmin=min, vmax=max)

def split_on_nan(x, y):
    """
    Split arrays (x, y) into contiguous segments, breaking wherever
    either x or y is NaN.

    Returns
    -------
    segments : list of (x_sub, y_sub)
        Each (x_sub, y_sub) is a contiguous valid part of the original arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    valid = ~(np.isnan(x) | np.isnan(y))

    if not np.any(valid):
        return []

    segments = []
    # Find the indices where validity changes
    idx = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
    # Add endpoints
    boundaries = np.concatenate(([0], idx, [len(x)]))

    # Collect segments where validity is True
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if valid[start]: # Only add segments that are valid
            segments.append((x[start:end], y[start:end]))
    return segments
