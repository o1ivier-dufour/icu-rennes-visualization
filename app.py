import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


st.set_page_config(page_title="UHI Rennes", layout="wide")
st.title("Urban heat island analysis in Rennes for 2018")


st.markdown(
    """
When I saw the visualizations on the
[Rennes Urban Climate Network (RUN)](https://run.letg.cnrs.fr/) website, particularly
the GIF showing the evolution of the urban heat island throughout the
day, I wondered what an interactive version of
these maps would look like: something that would allow users to choose the time, zoom in, and
explore the UHI's development in greater detail.

I retrieved the hourly temperature measurements for 2018
([dataset](https://www.easydata.earth/#/public/metadata/260c9e6b-d9d9-4559-b6ee-6e10ef1f7a13)).
Today, the network is much denser, but these data already provide
a good basis for exploring the dynamics of the UHI.

I focused on the period from June to August, when the contrasts between urban
and rural areas are most pronounced. To isolate the actual urban effect,
I divided the stations into two groups: urban and rural.


In the
[Dubreuil and al. (2020)](https://climatology.edpsciences.org/articles/climat/full_html/2020/01/climat20201706/climat20201706.html)
study, the UHI is calculated based on a single rural station (*Melesse*).

For my part, I chose to use the average of several stations
located on the edge of the densely urbanized area. This choice seemed
interesting to me in order to limit the potential impact of an isolated station
with particular characteristics (local shading, exposure, terrain configuration, etc.).

Before going any further, I checked the completeness of the measurements throughout
the summer. Rural stations generally provide good coverage, whichis essentia
l for a stable reference.  As for urban stations, completeness is more variable:
some havea lot of data, others a little less.

I have kept the entire network as it existed in 2018, naturally keeping
these limitations in mind.
"""
)

######  Data loading and basic configuration #####

@st.cache_data
def load_data():
    """Load station metadata and hourly temperature time series."""
    stations_df = pd.read_csv(
        "positions_stations_2018.csv",
        sep=";",
    )
    temp_long_geo = pd.read_csv(
        "temperature_stations_long.csv",
        sep=";",
        parse_dates=["datetime"],
    )
    temp_long_geo["temperature"] = pd.to_numeric(
        temp_long_geo["temperature"],
        errors="coerce",
    )
    return stations_df, temp_long_geo


stations, temp_long_geo = load_data()

rural_stations = [
    "Saint-Jacques",
    "Vezin-Coquet",
    "Melesse",
    "La-Lice",
    "Morinais",
    "St-Denis",
    "Mi-Foret",
]
all_stations = temp_long_geo["station_name"].unique()
urban_stations = [s for s in all_stations if s not in rural_stations]


###### Helper functions #####

lat0 = stations["lat"].mean()
lon0 = stations["long"].mean()


def deg_to_m(lon, lat, lon0=lon0, lat0=lat0):
    """Approximate degree -> meter conversion around Rennes."""
    dy = (lat - lat0) * 110_540.0
    dx = (lon - lon0) * (111_320.0 * np.cos(np.deg2rad(lat0)))
    return dx, dy


def m_to_deg(dx, dy, lon0=lon0, lat0=lat0):
    """Inverse of deg_to_m (meter -> degree)."""
    lat = dy / 110_540.0 + lat0
    lon = dx / (111_320.0 * np.cos(np.deg2rad(lat0))) + lon0
    return lon, lat


def create_UHI_overlay_image(grid_lon, grid_lat, Z):
    """
    Create a transparent UHI PNG with contours,
    to be used as an overlay layer in Mapbox.
    """
    min_lon, max_lon = grid_lon.min(), grid_lon.max()
    min_lat, max_lat = grid_lat.min(), grid_lat.max()

    cmap = LinearSegmentedColormap.from_list(
        "strong_rdbu",
        ["#053061", "#2166ac", "#f7f7f7", "#b2182b", "#67001f"],
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    Z_vis = np.clip(Z * 1.3, -3, 3)

    ax.contourf(
        grid_lon,
        grid_lat,
        Z_vis,
        levels=np.linspace(-3, 3, 60),
        cmap=cmap,
        vmin=-3,
        vmax=3,
        alpha=0.85,
        extend="both",
    )

    contours = ax.contour(
        grid_lon,
        grid_lat,
        Z,
        levels=np.linspace(-3, 3, 13),
        colors="black",
        linewidths=0.8,
        alpha=0.8,
    )
    ax.clabel(
        contours,
        inline=True,
        fontsize=7,
        fmt="%.1f°C",
        colors="black",
    )

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="none",
    )
    plt.close(fig)

    img_uri = (
        "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode("ascii")
    )
    coords = [
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]
    return img_uri, coords


def create_UHI_overlay_image_cumulative(grid_lon, grid_lat, Z):
    """
    Create a smoother UHI PNG overlay for cumulative maps
    (no isolines, soft colormap, 0–3 °C clipping).
    """
    min_lon, max_lon = grid_lon.min(), grid_lon.max()
    min_lat, max_lat = grid_lat.min(), grid_lat.max()

    cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    Z_clipped = np.ma.masked_invalid(np.clip(Z, 0, 3))

    ax.contourf(
        grid_lon,
        grid_lat,
        Z_clipped,
        levels=np.linspace(0, 3, 40),
        cmap=cmap,
        vmin=0,
        vmax=3,
        alpha=0.8,
        extend="max",
    )

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="none",
    )
    plt.close(fig)

    img_uri = (
        "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode("ascii")
    )
    coords = [
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]
    return img_uri, coords



##### Hottest days and completeness #####

@st.cache_data
def find_hottest_days(data, rural_station_list, n_days=10):
    """
    Select the n hottest days based on the mean of
    daily Tmax over rural stations.
    """
    rural_data = data[data["station_name"].isin(rural_station_list)].copy()
    rural_data["date"] = rural_data["datetime"].dt.date

    max_per_day = (
        rural_data.groupby(["date", "station_name"])["temperature"]
        .max()
        .reset_index()
    )
    mean_max_per_day = (
        max_per_day.groupby("date")["temperature"].mean().reset_index()
    )

    hottest_days = (
        mean_max_per_day.nlargest(n_days, "temperature")["date"].tolist()
    )
    return hottest_days


def calculate_completeness(data, period_start, period_end):
    """
    Compute hourly data completeness per station over a given period.
    """
    full_period = pd.date_range(start=period_start, end=period_end, freq="H")
    total_hours = len(full_period)
    results = []

    for station_name in all_stations:
        station_data = data[data["station_name"] == station_name]
        station_period = station_data[
            (station_data["datetime"] >= period_start)
            & (station_data["datetime"] <= period_end)
        ]
        available_hours = len(station_period.dropna(subset=["temperature"]))
        completeness = (available_hours / total_hours) * 100 if total_hours > 0 else 0

        results.append(
            {
                "Station": station_name,
                "Type": (
                    "Rural"
                    if station_name in rural_stations
                    else "Urban"
                ),
                "Taux_complétude": completeness,
            }
        )

    return pd.DataFrame(results)


def plot_presence_heatmap(presence_df, title, colorscale, col):
    """
    Display a boolean day/station presence matrix
    as a heatmap with a simple colorbar.
    """
    if presence_df.empty:
        return
    fig = px.imshow(
        presence_df.astype(int),
        title=title,
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=1,
        labels=dict(color="Presence"),
        aspect="auto",
    )
    fig.update_coloraxes(
        colorbar=dict(
            title="Mesure",
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Absent", "Present"],
        )
    )
    fig.update_traces(xgap=1, ygap=1)
    with col:
        st.plotly_chart(fig, use_container_width=True)


hottest_days = find_hottest_days(temp_long_geo, rural_stations, 10)

summer_stats = calculate_completeness(
    temp_long_geo,
    "2018-06-01",
    "2018-08-31",
)
summer_stats = summer_stats.sort_values(
    "Taux_complétude",
    ascending=False,
)

rural_stats = summer_stats[summer_stats["Type"] == "Rurale"].reset_index(
    drop=True
)
urban_stats = summer_stats[summer_stats["Type"] == "Urbaine"].reset_index(
    drop=True
)

rural_stats = rural_stats.drop(columns=["Type"])
urban_stats = urban_stats.drop(columns=["Type"])

summer_data = temp_long_geo[
    (temp_long_geo["datetime"] >= "2018-06-01")
    & (temp_long_geo["datetime"] <= "2018-08-31")
].copy()
summer_data["date"] = summer_data["datetime"].dt.date
daily_presence = (
    summer_data.groupby(["station_name", "date"])["temperature"]
    .count()
    .unstack(fill_value=0)
)
daily_presence = daily_presence > 0

rural_presence = daily_presence[daily_presence.index.isin(rural_stations)]
urban_presence = daily_presence[~daily_presence.index.isin(rural_stations)]


##### Display: completeness tables #####

col_rural, col_urban = st.columns(2)

with col_rural:
    st.subheader("Stations rurales")
    st.dataframe(
        rural_stats.style.format({"Completude_rate": "{:.1f}%"}),
        use_container_width=True,
    )

with col_urban:
    st.subheader("Stations urbaines")
    st.dataframe(
        urban_stats.style.format({"Completude_rate": "{:.1f}%"}),
        use_container_width=True,
    )

#### Display: daily presence figures #####

col1, col2 = st.columns(2)
plot_presence_heatmap(
    rural_presence,
    title="Rural stations",
    colorscale=[[0, "white"], [1, "blue"]],
    col=col1,
)
plot_presence_heatmap(
    urban_presence,
    title="Urban Stations",
    colorscale=[[0, "white"], [1, "green"]],
    col=col2,
)


##### Temperature map and UHI map for hottest day #####

st.markdown(
    """
To get an initial spatial overview, I focused on a single
summer day: the day when rural stations record their highest average
maximum temperatures. In other words, for each
day, I look at the maximum reached at each rural station and take
the average; the day chosen is the one with the highest
average.

This choice allows me to select a day that is truly hot on a regional scale,
based on the underlying air mass rather than local effects.

On this day, I then display the temperatures measured at the stations
on a map, with a slider to track the evolution hour by hour.
"""
)

hottest_day = hottest_days[0]
example_data = temp_long_geo[temp_long_geo["datetime"].dt.date == hottest_day]

available_hours = sorted(example_data["datetime"].dt.hour.unique())
selected_hour = st.slider(
    "Time of day",
    min_value=int(min(available_hours)),
    max_value=int(max(available_hours)),
    value=14,
    step=1,
    format="%dh",
    key="hour_slider",
)

target_data = example_data[example_data["datetime"].dt.hour == selected_hour]
if len(target_data) == 0:
    target_data = example_data

map_data = pd.merge(
    stations,
    target_data[["station_name", "temperature"]],
    on="station_name",
    how="left",
)

temp_map = go.Figure()

stations_with_temp = map_data.dropna(subset=["temperature"])

if len(stations_with_temp) > 0:
    temp_map.add_trace(
        go.Scattermapbox(
            lat=stations_with_temp["lat"],
            lon=stations_with_temp["long"],
            mode="markers",
            marker=dict(
                size=15,
                color=stations_with_temp["temperature"],
                reversescale=True,
                colorscale="hot",
                showscale=True,
                colorbar=dict(title="Température (°C)"),
            ),
            showlegend=False,
            text=(
                stations_with_temp["station_name"]
                + ": "
                + stations_with_temp["temperature"].round(1).astype(str)
                + "°C"
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

missing_data = map_data[map_data["temperature"].isna()]

urban_missing = missing_data[missing_data["station_name"].isin(urban_stations)]
if len(urban_missing) > 0:
    temp_map.add_trace(
        go.Scattermapbox(
            lat=urban_missing["lat"],
            lon=urban_missing["long"],
            mode="text",
            text=["×"] * len(urban_missing),
            textfont=dict(size=20, color="red"),
            showlegend=False,
            textposition="middle center",
            hovertext=(
                urban_missing["station_name"] + " (missing data)"
            ),
        )
    )

rural_missing = missing_data[missing_data["station_name"].isin(rural_stations)]
if len(rural_missing) > 0:
    temp_map.add_trace(
        go.Scattermapbox(
            lat=rural_missing["lat"],
            lon=rural_missing["long"],
            mode="text",
            text=["×"] * len(rural_missing),
            textfont=dict(size=20, color="black"),
            showlegend=False,
            textposition="middle center",
            hovertext=(
                rural_missing["station_name"] + " (missing data)"
            ),
        )
    )

avg_lat = stations["lat"].mean()
avg_lon = stations["long"].mean()

temp_map.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=10),
    height=420,
    title=f"Temperatures - {hottest_day} à {selected_hour:02d}h",
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
)

col_map, _ = st.columns([3, 1])
with col_map:
    st.plotly_chart(temp_map, use_container_width=True)

st.markdown(
    r"""
To convert raw temperatures to the urban effect, I calculate for
each station *i* and each hour *t*:

$$
UHI_i(t) = T_i(t) - \overline{T}_{\text{rural}}(t)
$$

where $\overline{T}_{\text{rural}}(t)$ is the average temperature of
rural stations at time *t*.

This first UHI map follows the logic of the temperature map
above, but applied to UHI values.  
This is not yet a continuous representation; it is simply
an intermediate step before interpolation, which we will perform immediately
afterwards.
"""
)

rural_with_temp = stations_with_temp[
    stations_with_temp["station_name"].isin(rural_stations)
]
if len(rural_with_temp) > 0:
    rural_ref_temp = rural_with_temp["temperature"].mean()

    stations_with_uhi = stations_with_temp.copy()
    stations_with_uhi["UHI"] = (
        stations_with_uhi["temperature"] - rural_ref_temp
    )

    uhi_map = go.Figure()

    uhi_map.add_trace(
        go.Scattermapbox(
            lat=stations_with_uhi["lat"],
            lon=stations_with_uhi["long"],
            mode="markers",
            marker=dict(
                size=15,
                color=stations_with_uhi["UHI"],
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="UHI (°C)"),
            ),
            showlegend=False,
            text=(
                stations_with_uhi["station_name"]
                + ": UHI "
                + stations_with_uhi["UHI"].round(2).astype(str)
                + "°C"
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    if len(urban_missing) > 0:
        uhi_map.add_trace(
            go.Scattermapbox(
                lat=urban_missing["lat"],
                lon=urban_missing["long"],
                mode="text",
                text=["×"] * len(urban_missing),
                textfont=dict(size=20, color="red"),
                showlegend=False,
                textposition="middle center",
                hovertext=(
                    urban_missing["station_name"] + " (missing data)"
                ),
            )
        )

    if len(rural_missing) > 0:
        uhi_map.add_trace(
            go.Scattermapbox(
                lat=rural_missing["lat"],
                lon=rural_missing["long"],
                mode="text",
                text=["×"] * len(rural_missing),
                textfont=dict(size=20, color="black"),
                showlegend=False,
                textposition="middle center",
                hovertext=(
                    rural_missing["station_name"] + " (missing data)"
                ),
            )
        )

    uhi_map.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=10),
        height=420,
        title=f"UHI - {hottest_day} à {selected_hour:02d}h",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )

    col_uhi, _ = st.columns([3, 1])
    with col_uhi:
        st.plotly_chart(uhi_map, use_container_width=True)


##### RBF UHI animation #####

st.markdown(
    r"""
To obtain a more continuous representation of the UHI in
space, I used scikit-learn interpolation based on a
Gaussian Process with an RBF kernel. This approach produces a smoothed field,
where the influence of a station decreases with distance. The kernel used
is:

$$
k(x,x') = \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right) + \sigma_n^2
$$

Here, $x$ denotes the position of a station (where the UHI is measured), and $x'$
the position where we want to estimate the UHI on the interpolated grid. The
parameter $\ell$ controls the distance over which the UHI varies, and
$\sigma_n^2$ adds a slight noise to account for local differences
between stations.
"""
)

st.markdown(
    """
Visually, the result is interesting: the interpolation reveals
a structure consistent with what one would expect from an urban heat island,
with contrasts that are clearly visible in
space.

Looking more closely: in the middle of the night, the heat island
appears clearly marked; in the early morning, the contrasts gradually
diminish; then, in the middle of the afternoon, we again observe a
more concentrated hot zone, less diffuse than during the night.

This overall consistency is encouraging, but the data must still be
interpreted with caution: as the network of stations is not very dense,
interpolation gives an idea of spatial contrasts without being able to
capture local variations in detail.
"""
)


@st.cache_data
def compute_all_frames():
    """
    Build UHI interpolation frames for each hour of the hottest day,
    and prepare Mapbox-ready image overlays.
    """
    uhi_frames = []
    timestamps = sorted(example_data["datetime"].unique())

    urban_df = stations[stations["station_name"].isin(urban_stations)]
    margin_deg = 0.03

    lon_min = urban_df["long"].min() - margin_deg
    lon_max = urban_df["long"].max() + margin_deg
    lat_min = urban_df["lat"].min() - margin_deg
    lat_max = urban_df["lat"].max() + margin_deg

    min_x, min_y = deg_to_m(lon_min, lat_min)
    max_x, max_y = deg_to_m(lon_max, lat_max)

    for timestamp in timestamps:
        time_slice = example_data[example_data["datetime"] == timestamp].copy()

        rural_temps = time_slice[
            time_slice["station_name"].isin(rural_stations)
        ]["temperature"]
        if len(rural_temps) == 0:
            continue

        rural_ref_temp = rural_temps.mean()
        time_slice["UHI"] = time_slice["temperature"] - rural_ref_temp

        slice_with_uhi = time_slice.dropna(subset=["UHI"]).copy()
        if len(slice_with_uhi) < 3:
            continue

        dx, dy = deg_to_m(
            slice_with_uhi["long"].values,
            slice_with_uhi["lat"].values,
        )
        X = np.column_stack([dx, dy])
        y = slice_with_uhi["UHI"].values

        kernel = RBF(length_scale=500.0) + WhiteKernel(noise_level=0.05)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=0,
        )
        gpr.fit(X, y)

        nx, ny = 250, 250
        gx, gy = np.meshgrid(
            np.linspace(min_x, max_x, nx),
            np.linspace(min_y, max_y, ny),
        )
        X_grid = np.column_stack([gx.ravel(), gy.ravel()])
        y_pred, _ = gpr.predict(X_grid, return_std=True)
        Z = y_pred.reshape(gx.shape)

        grid_lon, grid_lat = m_to_deg(gx, gy)
        img_uri, img_coords = create_uhi_overlay_image(
            grid_lon,
            grid_lat,
            Z,
        )

        uhi_frames.append(
            dict(
                instant=timestamp,
                slice_ok=slice_with_uhi,
                img_uri=img_uri,
                img_coords=img_coords,
                grid_lon=grid_lon,
                grid_lat=grid_lat,
                Z=Z,
            )
        )

    return uhi_frames


uhi_frames = compute_all_frames()

first_frame = uhi_frames[0]

urban_df = stations[stations["station_name"].isin(urban_stations)]
center_lat = float(urban_df["lat"].mean())
center_lon = float(urban_df["long"].mean())

uhi_anim = go.Figure()

uhi_anim.add_trace(
    go.Scattermapbox(
        lat=first_frame["slice_ok"]["lat"],
        lon=first_frame["slice_ok"]["long"],
        mode="markers",
        marker=dict(size=15, color="#2c3e50"),
        text=first_frame["slice_ok"]["station_name"],
        customdata=first_frame["slice_ok"]["UHI"],
        hovertemplate="<b>%{text}</b><br>UHI: %{customdata:.2f}°C<extra></extra>",
        name="Stations UHI",
    )
)

plotly_frames = []
for frame_data in uhi_frames:
    plotly_frames.append(
        go.Frame(
            data=[
                go.Scattermapbox(
                    lat=frame_data["slice_ok"]["lat"],
                    lon=frame_data["slice_ok"]["long"],
                    mode="markers",
                    marker=dict(size=15, color="#2c3e50"),
                    text=frame_data["slice_ok"]["station_name"],
                    customdata=frame_data["slice_ok"]["UHI"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "UHI: %{customdata:.2f}°C<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                mapbox=dict(
                    layers=[
                        dict(
                            sourcetype="image",
                            source=frame_data["img_uri"],
                            coordinates=frame_data["img_coords"],
                            opacity=0.8,
                        )
                    ]
                ),
                title=(
                    "Interpolate UHI – "
                    f"{frame_data['instant'].strftime('%d/%m/%Y %Hh%M')}"
                ),
            ),
            name=str(frame_data["instant"].hour),
        )
    )

uhi_anim.frames = plotly_frames

time_slider = [
    dict(
        active=0,
        currentvalue=dict(prefix="<b>Hour : </b>", font=dict(size=16)),
        pad=dict(t=60, b=10),
        len=0.85,
        x=0.12,
        xanchor="left",
        steps=[
            dict(
                args=[
                    [str(frame["instant"].hour)],
                    dict(
                        frame=dict(duration=600, redraw=True),
                        mode="immediate",
                        transition=dict(duration=300),
                    ),
                ],
                label=f"{frame['instant'].hour:02d}h",
                method="animate",
            )
            for frame in uhi_frames
        ],
    )
]

animation_buttons = [
    dict(
        type="buttons",
        showactive=False,
        x=0.01,
        y=0.02,
        xanchor="left",
        yanchor="bottom",
        direction="right",
        pad=dict(r=10, t=10),
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=900, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                        transition=dict(duration=400),
                    ),
                ],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ],
    )
]

uhi_anim.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=11,
        layers=[
            dict(
                sourcetype="image",
                source=first_frame["img_uri"],
                coordinates=first_frame["img_coords"],
                opacity=0.8,
            )
        ],
    ),
    sliders=time_slider,
    updatemenus=animation_buttons,
    height=550,
    margin=dict(l=10, r=10, t=90, b=60),
    title=dict(
        text=(
            "<b>Urban Heat Islands – Rennes</b>"
            f"<br><sub>{hottest_day}</sub>"
        ),
        x=0.5,
        xanchor="center",
    ),
    showlegend=False,
)

col_anim, _ = st.columns([3, 1])
with col_anim:
    st.plotly_chart(uhi_anim, use_container_width=True)


##### Cumulative UHI maps (day / night) #####

st.markdown(
    r"""
Beyond the hourly maps, it may also be useful to identify the
areas that remain consistently hot, and not just those that
appear intense at a specific time. Hot spots are not necessarily located
in the same place between day and night, because the mechanisms at
play are not the same: during the day, sunlight and surface heating
dominate, while in the middle of the night, it is mainly the
release of stored heat that structures the heat island. Added to this
are local factors such as ventilation and urban morphology,
 which can also shift the maxima.

To identify more persistent areas, I first separated
daytime and nighttime hours, then chose to combine the UHIs interpolated
over the ten hottest days. Ten days seemed to me to be a good order of
magnitude: enough to smooth out day-to-day variations without
going too far toward an average that is too broad or diluted over time.

For each point $x$ on the grid, I then calculate an average over
all the selected time points:

$$
UHI_{\text{moy}}(x) = \frac{1}{N} \sum_{t=1}^{N} UHI_t(x)
$$

where $UHI_t(x)$ is the UHI interpolated at point $x$ at time $t$, and $N$ is
the total number of hours considered (day or night) on the hottest days.
This average highlights areas where the UHI is both high and recurrent,
in other words, typical hot spots during the day and night..

These maps primarily provide a clear visual snapshot, which helps identify
the areas most frequently exposed. However, with the current network of stations
being much more comprehensive, this type of representation could
become highly informative, to the point of providing a solid basis for
supporting certain decisions.
"""
)

st.markdown(r"""
**Analysis – Daytime UHI**

During the day, the UHI remains generally weak, which is consistent with
the literature: urban/rural contrasts are limited as long as the thermal inertia
of the city is not fully effective and heat release has not yet begun. 
However, the city center stands out clearly, in line with conventional observations.

Two small areas also appear, notably above Gayeulles Park, which is less intuitive for a green area.
This is probably due to the low density of the network: the position of the stations influences
the interpolation locally and can reveal patterns that do not always perfectly reflect reality.

Overall, the reading is consistent for the city center, but certain peripheral details should be interpreted with caution, given the limited number of measurement points.
""")

day_hours = list(range(9, 19))
night_hours = list(range(0, 6))


def interpolate_daily_data(valid_data, urban_bbox):
    """
    Interpolate UHI values at a given time over the
    urban bounding box using a Gaussian Process.
    """
    dx, dy = deg_to_m(
        valid_data["long"].values,
        valid_data["lat"].values,
    )
    X = np.column_stack([dx, dy])
    y = valid_data["UHI"].values

    kernel = RBF(length_scale=800.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        random_state=0,
    )
    gpr.fit(X, y)

    lon_min_m, lat_min_m = deg_to_m(
        urban_bbox["lon_min"],
        urban_bbox["lat_min"],
    )
    lon_max_m, lat_max_m = deg_to_m(
        urban_bbox["lon_max"],
        urban_bbox["lat_max"],
    )

    nx, ny = 100, 100
    gx, gy = np.meshgrid(
        np.linspace(lon_min_m, lon_max_m, nx),
        np.linspace(lat_min_m, lat_max_m, ny),
    )
    X_grid = np.column_stack([gx.ravel(), gy.ravel()])
    y_pred, _ = gpr.predict(X_grid, return_std=True)
    Z = y_pred.reshape(gx.shape)

    grid_lon, grid_lat = m_to_deg(gx, gy)
    return Z, grid_lon, grid_lat


@st.cache_data
def compute_spatial_cumulative_means(
    hottest_days,
    day_hours,
    night_hours,
):
    """
    Compute mean UHI fields for day and night on the n hottest days,
    by aggregating all selected hourly interpolations.
    """
    day_cumulative = None
    night_cumulative = None
    grid_lon_ref = None
    grid_lat_ref = None

    urban_stations_df = stations[stations["station_name"].isin(urban_stations)]

    margin = 0.03
    urban_bbox = {
        "lat_min": urban_stations_df["lat"].min() - margin,
        "lat_max": urban_stations_df["lat"].max() + margin,
        "lon_min": urban_stations_df["long"].min() - margin,
        "lon_max": urban_stations_df["long"].max() + margin,
    }

    # Daytime aggregation
    for day in hottest_days:
        for hour in day_hours:
            day_data = temp_long_geo[
                (temp_long_geo["datetime"].dt.date == day)
                & (temp_long_geo["datetime"].dt.hour == hour)
            ].copy()

            if len(day_data) < 3:
                continue

            day_rural_temps = day_data[
                day_data["station_name"].isin(rural_stations)
            ]["temperature"]
            if len(day_rural_temps) == 0:
                continue

            day_rural_ref_temp = day_rural_temps.mean()
            day_data["UHI"] = (
                day_data["temperature"] - day_rural_ref_temp
            )

            day_valid_data = day_data.dropna(subset=["UHI"]).copy()
            if len(day_valid_data) < 3:
                continue

            Z_day, grid_lon, grid_lat = interpolate_daily_data(
                day_valid_data,
                urban_bbox,
            )

            if day_cumulative is None:
                day_cumulative = Z_day.copy()
                grid_lon_ref = grid_lon
                grid_lat_ref = grid_lat
            else:
                if day_cumulative.shape == Z_day.shape:
                    day_cumulative += Z_day

    # Nighttime aggregation
    for day in hottest_days:
        for hour in night_hours:
            night_data = temp_long_geo[
                (temp_long_geo["datetime"].dt.date == day)
                & (temp_long_geo["datetime"].dt.hour == hour)
            ].copy()

            if len(night_data) < 3:
                continue

            night_rural_temps = night_data[
                night_data["station_name"].isin(rural_stations)
            ]["temperature"]
            if len(night_rural_temps) == 0:
                continue

            night_rural_ref_temp = night_rural_temps.mean()
            night_data["UHI"] = (
                night_data["temperature"] - night_rural_ref_temp
            )

            night_valid_data = night_data.dropna(subset=["UHI"]).copy()
            if len(night_valid_data) < 3:
                continue

            Z_night, _, _ = interpolate_daily_data(
                night_valid_data,
                urban_bbox,
            )

            if night_cumulative is None:
                night_cumulative = Z_night.copy()
            else:
                if night_cumulative.shape == Z_night.shape:
                    night_cumulative += Z_night

    day_cumulative_mean = None
    night_cumulative_mean = None

    if day_cumulative is not None:
        day_cumulative_mean = day_cumulative / max(
            1,
            len(day_hours) * len(hottest_days),
        )

    if night_cumulative is not None:
        night_cumulative_mean = night_cumulative / max(
            1,
            len(night_hours) * len(hottest_days),
        )

    return (
        day_cumulative_mean,
        night_cumulative_mean,
        grid_lon_ref,
        grid_lat_ref,
        urban_bbox,
    )


with st.spinner("Calculation of cumulative totals over several hours..."):
    (
        day_cumulative_mean,
        night_cumulative_mean,
        grid_lon,
        grid_lat,
        urban_bbox,
    ) = compute_spatial_cumulative_means(
        hottest_days,
        day_hours,
        night_hours,
    )

urban_stations_df = stations[stations["station_name"].isin(urban_stations)]
urban_center = {
    "lat": urban_stations_df["lat"].mean(),
    "lon": urban_stations_df["long"].mean(),
}


def plot_cumulative_overlay(
    Z_mean,
    grid_lon,
    grid_lat,
    title,
    stations_df,
    center,
):
    """
    Plot a mean UHI field as a Mapbox overlay with station markers
    and a separate colorbar.
    """
    Z_vis = Z_mean

    img_uri, img_coords = create_uhi_overlay_image_cumulative(
        grid_lon,
        grid_lat,
        Z_vis,
    )

    fig = go.Figure()

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=11,
            domain={"x": [0, 0.9], "y": [0, 1]},
            layers=[
                dict(
                    sourcetype="image",
                    source=img_uri,
                    coordinates=img_coords,
                    opacity=0.85,
                )
            ],
        ),
        height=480,
        margin=dict(l=0, r=80, t=60, b=0),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        showlegend=False,
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=stations_df["lat"],
            lon=stations_df["long"],
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.7),
            text=stations_df["station_name"],
            name="Urban stations",
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=[[0, 3]],
            x=[0, 1],
            y=[0, 1],
            colorscale="YlOrRd",
            showscale=True,
            opacity=0.0,
            hoverinfo="skip",
            colorbar=dict(
                title=dict(text="Average UHI (°C)", side="right"),
                x=1.02,
                xanchor="left",
                len=0.8,
            ),
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    col_cumulative, _ = st.columns([3, 1])
    with col_cumulative:
        st.plotly_chart(fig, use_container_width=True)


day_title = (
    f"Average UHI – DAY (heures {min(day_hours)}–{max(day_hours)}h, "
    f"{len(hottest_days)} hottest days)"
)
plot_cumulative_overlay(
    day_cumulative_mean,
    grid_lon,
    grid_lat,
    day_title,
    urban_stations_df,
    urban_center,
)

st.markdown(
    """
**Analysis – Night UHI**

At night, the UHI increases significantly, which is consistent with what
is described in the literature—notably the study by
[Dubreuil and al. (2020)](https://climatology.edpsciences.org/articles/climat/full_html/2020/01/climat20201706/climat20201706.html)
on summer nights in Rennes. Once solar radiation stops,
urban surfaces release the accumulated heat, and it is generally at
this point that the urban/rural contrast reaches its maximum.

On the average map, the center of the block remains firmly fixed on the hypercenter,
in line with the results presented in the article. We also observe
an extension towards the south, a pattern that fairly clearly reflects the
distribution of stations: unlike what happened during the
day, it is not an isolated point that guides the interpolation, but
several stations that pull the structure in this direction. With a
denser network, we would undoubtedly see a slightly more
pronounced extension to the north as well, which is already suggested in the literature.

The overall organization of the map is therefore consistent with what
is expected of a marked nocturnal heat island. The fine details should
still be considered with caution, but the overall balance appears solid 
given the number of stations available.
"""
)

night_title = (
    f"Average UHI – NIGHT (hours {min(night_hours)}–{max(night_hours)}h, "
    f"{len(hottest_days)} hottest days)"
)
plot_cumulative_overlay(
    night_cumulative_mean,
    grid_lon,
    grid_lat,
    night_title,
    urban_stations_df,
    urban_center,
)


st.markdown(
    """
**Conclusion**

This prototype provides an initial visualization of heat islands in
Rennes during the summer of 2018, focusing on the hottest days of the
June–August period. The maps obtained already provide a simple illustration
of the phenomenon: they highlight the day/night contrasts, reveal recurring
hot spots, and offer a clear visual overview that is useful for someone new
to the subject. The results are broadly consistent with those described
in the literature, which is encouraging.

In this work, I only used hourly temperatures from the23 stations available in 2018.
This is a subset of the RUN network, which now combines several types of sensors:
automaticmeasuring at least temperature, relative humidity, wind,rain, and pressure
(sometimes radiation), and a denser network of LoRaWAN sensors recording temperature
and humidity. Not all stations measure all of these variables, but the network still
provides a much richer basis than the one used here.

A natural next step would therefore be to use this additional data
to analyze more detailed combinations of factors:
for example, identifying the most unfavorable situations (low wind,
clear skies, low humidity, limited vegetation), or understanding how
the characteristics of buildings and vegetation actually structure
the UHI. These maps could also be cross-referenced with other sources: broader
meteorological fields, satellite imagery (vegetation indices, thermal), or
classifications of buildings and vegetation derived from IGN data.

As this information is combined, statistical approaches, machine learning,
or even deep learning could become relevant—not to replace the work
of urban planners, but to provide them with more powerful tools.
The idea would be for them to interact with the data themselves,
explore different settings, run their own analyses, and benefit from more
accurate indicators to support their choices.In the longer term, we could
even imagine a modeling component, where they could test different development
or adaptation scenarios to compare their effects on the heat island.In the longer
term, we can even imagine a modeling component, where they could test different
development or adaptation scenarios to compare their effects on the heat island.

The ultimate goal is to help make cities more livable during summer heat waves,
especially for the most vulnerable populations. We know that heat waves are
associated with significantly higher mortality rates, particularly among the elderly.
Having more accurate, interactive tools tailored to the needs of professionals could
help limit these impacts.

Beyond the operational aspect, this type of approach could also
be part of a broader research effort. By combining dense local data,
detailed geographic information, and more advanced analytical methods,
we could contribute to improving our understanding of urban heat islands
and, more broadly, issues related to climate change adaptation. The tools developed
could be reused or adapted by other communities, or even
serve as a basis for work carried out in other regions or countries,
promoting a more open and cumulative science of UHI.

In a context where populations are increasingly concentrated in cities
and heat waves are intensifying, having robust, shareable, and truly useful
methods would be a significant step forward. Without claiming to address 
all the issues, this type of approach could contribute, at its own level,
to improving knowledge anddesigning decision-making tools that are better
suited to the challenges ahead.
"""
)


