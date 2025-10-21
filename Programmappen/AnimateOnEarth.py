import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

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

        im = draw_earth(ax, res)

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

def draw_earth(ax, res):
    # draw circular outline
    circle = plt.Circle((0, 0), 1, color='k', lw=1.2, fill=False)
    ax.add_artist(circle)

    # add coastlines from file (computed by make_coastline_data)
    with open(os.path.join(os.path.dirname(__file__), 'coastline_data.pkl'), 'rb') as f:
        coastline_segments = pickle.load(f)
    for x_c, y_c in coastline_segments:
        ax.plot(x_c, y_c, color='black', lw=0.5)

    # colors = [[0, 0.7, 1, 1], [1, 1, 1, 1], [1, 0.2, 0, 1]] # blue to white to red
    # zero_point = np.clip(-min / (max - min), 0, 1)
    # cmap = ListedColormap(np.vstack((np.linspace(colors[0], colors[1], int(256*zero_point)),
    #                                 np.linspace(colors[1], colors[2], 256 - int(256*zero_point)))))
    min_temp, max_temp = -50, 50
    dmi_colors_cold_rgb = [[0, 30, 150], [71, 96, 160],[0, 143, 233], [61, 171, 238], [109, 191, 242],
                           [22, 225, 204], [125, 238, 226], [158, 242, 233], [255, 255, 255]]
    dmi_colors_warm_rgb = [[255, 255, 255], [255, 236, 127], [255, 217, 0], [255, 178, 0],
                            [255, 142, 82], [255, 181, 181], [255, 157, 157], [255, 124, 124],
                            [255, 82, 82], [230, 57, 57], [204, 31, 31], [128, 0, 0], [92, 0, 51]]
    dmi_colors_cold = np.array(dmi_colors_cold_rgb)/255.0
    dmi_colors_warm = np.array(dmi_colors_warm_rgb)/255.0
    cmap = ListedColormap(np.transpose([np.concatenate((np.interp(np.linspace(0, 1, 4*abs(min_temp)), np.linspace(0, 1, len(dmi_colors_cold)), dmi_colors_cold[:,i]),
                                    np.interp(np.linspace(0, 1, 4*max_temp), np.linspace(0, 1, len(dmi_colors_warm)), dmi_colors_warm[:,i])[1:]))
                                    for i in range(3)]))
    return ax.imshow(np.zeros((res,res)), extent=(-1, 1, -1, 1), origin='lower',
                cmap=cmap, interpolation='bilinear', vmin=min_temp, vmax=max_temp)

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

def make_coastline_data():
    """
    Create and save coastline data from Natural Earth shapefiles. Depends on cartopy which can be difficult to install.
    The data is saved in 'coastline_data.pkl' for later use in plotting.
    """
    import cartopy.io.shapereader as shpreader

    reader = shpreader.natural_earth(resolution='110m', category='physical', name='coastline') 
    segments = []
    for record in shpreader.Reader(reader).records():
        coords = np.array(record.geometry.coords)
        lon_coords, lat_coords = np.radians(coords[:,0]), np.radians(coords[:,1])
        behind = np.cos(lon_coords) < 0
        lon_coords[behind] = np.nan
        y_coords = np.sin(lat_coords)
        x_coords = np.cos(lat_coords) * np.sin(lon_coords)
        for x_coords_segment, y_coords_segment in split_on_nan(x_coords, y_coords):
            segments.append((x_coords_segment, y_coords_segment))
    
    with open('coastline_data.pkl', 'wb') as f:
        pickle.dump(segments, f)
    
    print(f"Saved {len(segments)} coastline segments to 'coastline_data.pkl'")