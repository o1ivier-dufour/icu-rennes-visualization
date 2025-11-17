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


st.set_page_config(page_title="ICU Rennes", layout="wide")
st.title("Analyse ICU Rennes 2018")


st.markdown(
    """
En voyant les visualisations proposées sur le site du
[Rennes Urban Climate Network (RUN)](https://run.letg.cnrs.fr/), notamment
le GIF qui montre l’évolution de l’îlot de chaleur urbain au fil de la
journée, je me suis demandé ce que donnerait une version interactive de
ces cartes : quelque chose qui permette de choisir l’heure, de zoomer et
d’explorer plus finement la mise en place de l’ICU.

J’ai récupéré les mesures horaires de température pour l’année 2018
([dataset](https://www.easydata.earth/#/public/metadata/260c9e6b-d9d9-4559-b6ee-6e10ef1f7a13)).
Aujourd’hui, le réseau est beaucoup plus dense, mais ces données offrent
déjà une bonne base pour explorer la dynamique de l’ICU.

Je me suis concentré sur la période de juin à août, là où les contrastes
urbain/rural sont les plus marqués. Pour isoler l’effet réellement urbain,
j’ai séparé les stations en deux groupes : urbaines et rurales.

Dans l’étude de
[Dubreuil et al. (2020)](https://climatology.edpsciences.org/articles/climat/full_html/2020/01/climat20201706/climat20201706.html),
l’ICU est calculé à partir d’une station rurale unique (*Melesse*).

De mon côté, j’ai choisi d’utiliser la moyenne de plusieurs stations
situées en marge de la zone densément urbanisée. Ce choix me semblait
intéressant pour limiter l’impact potentiel d’une station isolée
présentant un comportement particulier (ombrage local, exposition,
configuration du terrain…).

Avant d’aller plus loin, j’ai vérifié la complétude des mesures sur tout
l’été. Les stations rurales présentent globalement une bonne couverture, ce qui
est essentiel pour disposer d’une référence stable.  
Côté stations urbaines, la complétude est plus variable : certaines ont
beaucoup de données, d’autres un peu moins.

J’ai conservé l’ensemble du réseau tel qu’il existait en 2018, en gardant
naturellement ces limites en tête.
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


def create_icu_overlay_image(grid_lon, grid_lat, Z):
    """
    Create a transparent ICU PNG with contours,
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


def create_icu_overlay_image_cumulative(grid_lon, grid_lat, Z):
    """
    Create a smoother ICU PNG overlay for cumulative maps
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
                    "Rurale"
                    if station_name in rural_stations
                    else "Urbaine"
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
        labels=dict(color="Présence"),
        aspect="auto",
    )
    fig.update_coloraxes(
        colorbar=dict(
            title="Mesure",
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Absente", "Présente"],
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
        rural_stats.style.format({"Taux_complétude": "{:.1f}%"}),
        use_container_width=True,
    )

with col_urban:
    st.subheader("Stations urbaines")
    st.dataframe(
        urban_stats.style.format({"Taux_complétude": "{:.1f}%"}),
        use_container_width=True,
    )

#### Display: daily presence figures #####

col1, col2 = st.columns(2)
plot_presence_heatmap(
    rural_presence,
    title="Stations rurales",
    colorscale=[[0, "white"], [1, "blue"]],
    col=col1,
)
plot_presence_heatmap(
    urban_presence,
    title="Stations urbaines",
    colorscale=[[0, "white"], [1, "green"]],
    col=col2,
)


##### Temperature map and ICU map for hottest day #####

st.markdown(
    """
Pour avoir un premier aperçu spatial, je me suis concentré sur une seule
journée d’été : celle où les stations rurales enregistrent en moyenne
leurs températures maximales les plus élevées. Autrement dit, pour chaque
jour, je regarde le maximum atteint dans chaque station rurale et j’en
prends la moyenne ; le jour retenu est celui où cette moyenne est la plus
forte.

Ce choix permet de sélectionner une journée réellement chaude à l’échelle
régionale, en se basant sur la masse d’air de fond plutôt que sur des
effets locaux.

Sur cette journée, j’affiche ensuite les températures mesurées aux
stations sur une carte, avec un slider pour suivre l’évolution heure par
heure.
"""
)

hottest_day = hottest_days[0]
example_data = temp_long_geo[temp_long_geo["datetime"].dt.date == hottest_day]

available_hours = sorted(example_data["datetime"].dt.hour.unique())
selected_hour = st.slider(
    "Heure de la journée",
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
                urban_missing["station_name"] + " (donnée manquante)"
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
                rural_missing["station_name"] + " (donnée manquante)"
            ),
        )
    )

avg_lat = stations["lat"].mean()
avg_lon = stations["long"].mean()

temp_map.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=10),
    height=420,
    title=f"Températures - {hottest_day} à {selected_hour:02d}h",
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
)

col_map, _ = st.columns([3, 1])
with col_map:
    st.plotly_chart(temp_map, use_container_width=True)

st.markdown(
    r"""
Pour passer des températures brutes à l’effet urbain, je calcule pour
chaque station *i* et chaque heure *t* :

$$
ICU_i(t) = T_i(t) - \overline{T}_{\text{rural}}(t)
$$

où $\overline{T}_{\text{rural}}(t)$ est la moyenne des températures des
stations rurales à l’heure *t*.

Cette première carte d’ICU reprend la logique de la carte de température
ci-dessus, mais appliquée aux valeurs d’ICU.  
Il ne s’agit pas encore d’une représentation continue ; c’est simplement
une étape intermédiaire avant l’interpolation, que l’on va réaliser juste
après.
"""
)

rural_with_temp = stations_with_temp[
    stations_with_temp["station_name"].isin(rural_stations)
]
if len(rural_with_temp) > 0:
    rural_ref_temp = rural_with_temp["temperature"].mean()

    stations_with_icu = stations_with_temp.copy()
    stations_with_icu["ICU"] = (
        stations_with_icu["temperature"] - rural_ref_temp
    )

    icu_map = go.Figure()

    icu_map.add_trace(
        go.Scattermapbox(
            lat=stations_with_icu["lat"],
            lon=stations_with_icu["long"],
            mode="markers",
            marker=dict(
                size=15,
                color=stations_with_icu["ICU"],
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="ICU (°C)"),
            ),
            showlegend=False,
            text=(
                stations_with_icu["station_name"]
                + ": ICU "
                + stations_with_icu["ICU"].round(2).astype(str)
                + "°C"
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    if len(urban_missing) > 0:
        icu_map.add_trace(
            go.Scattermapbox(
                lat=urban_missing["lat"],
                lon=urban_missing["long"],
                mode="text",
                text=["×"] * len(urban_missing),
                textfont=dict(size=20, color="red"),
                showlegend=False,
                textposition="middle center",
                hovertext=(
                    urban_missing["station_name"] + " (donnée manquante)"
                ),
            )
        )

    if len(rural_missing) > 0:
        icu_map.add_trace(
            go.Scattermapbox(
                lat=rural_missing["lat"],
                lon=rural_missing["long"],
                mode="text",
                text=["×"] * len(rural_missing),
                textfont=dict(size=20, color="black"),
                showlegend=False,
                textposition="middle center",
                hovertext=(
                    rural_missing["station_name"] + " (donnée manquante)"
                ),
            )
        )

    icu_map.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=10),
        height=420,
        title=f"ICU - {hottest_day} à {selected_hour:02d}h",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )

    col_icu, _ = st.columns([3, 1])
    with col_icu:
        st.plotly_chart(icu_map, use_container_width=True)


##### RBF ICU animation #####

st.markdown(
    r"""
Pour obtenir ensuite une représentation plus continue de l’ICU dans
l’espace, j’ai utilisé l’interpolation de scikit-learn basée sur un
Gaussian Process avec un noyau RBF. Cette approche produit un champ lissé,
où l’influence d’une station décroît avec la distance. Le noyau utilisé
est :

$$
k(x,x') = \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right) + \sigma_n^2
$$

Ici, $x$ désigne la position d’une station (où l’ICU est mesuré), et $x'$
la position où l’on souhaite estimer l’ICU sur la grille interpolée. Le
paramètre $\ell$ contrôle la distance sur laquelle l’ICU varie, et
$\sigma_n^2$ ajoute un léger bruit pour tenir compte des écarts locaux
entre stations.
"""
)

st.markdown(
    """
Visuellement, le résultat est intéressant : l’interpolation fait apparaître
une structure cohérente avec ce que l’on attend d’un îlot de chaleur
urbain, avec des contrastes qui se dessinent de manière lisible dans
l’espace.

Si l’on regarde de plus près : au cœur de la nuit, l’îlot de chaleur
apparaît nettement marqué ; au petit matin, les contrastes se réduisent
progressivement ; puis, en milieu d’après-midi, on observe à nouveau une
zone chaude plus concentrée, moins diffuse que durant la nuit.

Cette cohérence générale est encourageante, mais l’ensemble doit tout de
même être interprété avec précaution : le réseau de stations étant peu
dense, l’interpolation donne une idée des contrastes spatiaux, sans pouvoir
capturer finement les variations locales.
"""
)


@st.cache_data
def compute_all_frames():
    """
    Build ICU interpolation frames for each hour of the hottest day,
    and prepare Mapbox-ready image overlays.
    """
    icu_frames = []
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
        time_slice["ICU"] = time_slice["temperature"] - rural_ref_temp

        slice_with_icu = time_slice.dropna(subset=["ICU"]).copy()
        if len(slice_with_icu) < 3:
            continue

        dx, dy = deg_to_m(
            slice_with_icu["long"].values,
            slice_with_icu["lat"].values,
        )
        X = np.column_stack([dx, dy])
        y = slice_with_icu["ICU"].values

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
        img_uri, img_coords = create_icu_overlay_image(
            grid_lon,
            grid_lat,
            Z,
        )

        icu_frames.append(
            dict(
                instant=timestamp,
                slice_ok=slice_with_icu,
                img_uri=img_uri,
                img_coords=img_coords,
                grid_lon=grid_lon,
                grid_lat=grid_lat,
                Z=Z,
            )
        )

    return icu_frames


icu_frames = compute_all_frames()

first_frame = icu_frames[0]

urban_df = stations[stations["station_name"].isin(urban_stations)]
center_lat = float(urban_df["lat"].mean())
center_lon = float(urban_df["long"].mean())

icu_anim = go.Figure()

icu_anim.add_trace(
    go.Scattermapbox(
        lat=first_frame["slice_ok"]["lat"],
        lon=first_frame["slice_ok"]["long"],
        mode="markers",
        marker=dict(size=15, color="#2c3e50"),
        text=first_frame["slice_ok"]["station_name"],
        customdata=first_frame["slice_ok"]["ICU"],
        hovertemplate="<b>%{text}</b><br>ICU: %{customdata:.2f}°C<extra></extra>",
        name="Stations ICU",
    )
)

plotly_frames = []
for frame_data in icu_frames:
    plotly_frames.append(
        go.Frame(
            data=[
                go.Scattermapbox(
                    lat=frame_data["slice_ok"]["lat"],
                    lon=frame_data["slice_ok"]["long"],
                    mode="markers",
                    marker=dict(size=15, color="#2c3e50"),
                    text=frame_data["slice_ok"]["station_name"],
                    customdata=frame_data["slice_ok"]["ICU"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "ICU: %{customdata:.2f}°C<extra></extra>"
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
                    "ICU interpolé – "
                    f"{frame_data['instant'].strftime('%d/%m/%Y %Hh%M')}"
                ),
            ),
            name=str(frame_data["instant"].hour),
        )
    )

icu_anim.frames = plotly_frames

time_slider = [
    dict(
        active=0,
        currentvalue=dict(prefix="<b>Heure : </b>", font=dict(size=16)),
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
            for frame in icu_frames
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

icu_anim.update_layout(
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
            "<b>Îlots de chaleur urbains – Rennes</b>"
            f"<br><sub>{hottest_day}</sub>"
        ),
        x=0.5,
        xanchor="center",
    ),
    showlegend=False,
)

col_anim, _ = st.columns([3, 1])
with col_anim:
    st.plotly_chart(icu_anim, use_container_width=True)


##### Cumulative ICU maps (day / night) #####

st.markdown(
    r"""
Au-delà des cartes horaires, il peut aussi être intéressant de repérer les
zones qui restent régulièrement chaudes, et pas seulement celles qui
apparaissent fortes à un moment précis. Les points chauds ne se situent pas
forcément au même endroit entre le jour et la nuit, car les mécanismes en
jeu ne sont pas les mêmes : en journée, l’ensoleillement et le chauffage
des surfaces dominent, alors qu’au cœur de la nuit, c’est surtout la
restitution de la chaleur stockée qui structure l’îlot de chaleur. À cela
s’ajoutent des facteurs locaux comme la ventilation ou la morphologie
urbaine, qui peuvent également déplacer les maxima.

Pour approcher des zones plus persistantes, j’ai donc d’abord séparé les
heures de jour et de nuit, puis j’ai choisi de cumuler les ICU interpolés
sur les dix jours les plus chauds. Dix jours me paraissaient un bon ordre
de grandeur : assez pour lisser les variations d’un jour à l’autre, sans
aller vers une moyenne trop large ou diluée dans le temps.

Pour chaque point $x$ de la grille, je calcule ensuite une moyenne sur
l’ensemble des instants retenus :

$$
ICU_{\text{moy}}(x) = \frac{1}{N} \sum_{t=1}^{N} ICU_t(x)
$$

où $ICU_t(x)$ est l’ICU interpolé au point $x$ à l’instant $t$, et $N$ le
nombre total d’heures considérées (jour ou nuit) sur les jours les plus
chauds. Cette moyenne permet de faire ressortir les zones où l’ICU est à la
fois élevé et récurrent, autrement dit les points chauds typiques du jour
et de la nuit.

Ces cartes donnent surtout un instantané visuel clair, qui aide à repérer
les secteurs les plus souvent exposés. Mais avec le réseau de stations
actuel qui est bien plus complet, ce type de représentation pourrait
devenir très informatif, au point de constituer une base solide pour
appuyer certaines décisions.
"""
)

st.markdown(r"""
**Analyse – ICU de jour**

En journée, l’ICU reste globalement faible, ce qui correspond bien à la littérature : les contrastes urbain/rural sont limités tant que l’inertie thermique de la ville ne joue pas pleinement et que la restitution de chaleur n’a pas encore commencé. L’hypercentre ressort néanmoins clairement, en cohérence avec les observations classiques.

Deux petites zones apparaissent aussi, notamment au-dessus du parc des Gayeulles, ce qui est moins intuitif pour un secteur végétalisé. Cela tient probablement à la faible densité du réseau : la position des stations influence localement l’interpolation et peut faire émerger des motifs qui ne reflètent pas toujours parfaitement la réalité.

Dans l’ensemble, la lecture est cohérente sur le centre-ville, mais certains détails périphériques doivent être interprétés avec prudence, compte tenu du nombre limité de points de mesure.
""")

day_hours = list(range(9, 19))
night_hours = list(range(0, 6))


def interpolate_daily_data(valid_data, urban_bbox):
    """
    Interpolate ICU values at a given time over the
    urban bounding box using a Gaussian Process.
    """
    dx, dy = deg_to_m(
        valid_data["long"].values,
        valid_data["lat"].values,
    )
    X = np.column_stack([dx, dy])
    y = valid_data["ICU"].values

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
    Compute mean ICU fields for day and night on the n hottest days,
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
            day_data["ICU"] = (
                day_data["temperature"] - day_rural_ref_temp
            )

            day_valid_data = day_data.dropna(subset=["ICU"]).copy()
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
            night_data["ICU"] = (
                night_data["temperature"] - night_rural_ref_temp
            )

            night_valid_data = night_data.dropna(subset=["ICU"]).copy()
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


with st.spinner("Calcul des cumulatifs sur plusieurs heures..."):
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
    Plot a mean ICU field as a Mapbox overlay with station markers
    and a separate colorbar.
    """
    Z_vis = Z_mean

    img_uri, img_coords = create_icu_overlay_image_cumulative(
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
            name="Stations urbaines",
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
                title=dict(text="ICU moyen (°C)", side="right"),
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
    f"ICU moyen – JOUR (heures {min(day_hours)}–{max(day_hours)}h, "
    f"{len(hottest_days)} jours les plus chauds)"
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
**Analyse – ICU de nuit**

La nuit, l’ICU se renforce nettement, ce qui est cohérent avec ce que
décrit la littérature — notamment l’étude de
[Dubreuil et al.(2020)](https://climatology.edpsciences.org/articles/climat/full_html/2020/01/climat20201706/climat20201706.html)
sur les nuits d’été à Rennes. Une fois le rayonnement solaire arrêté, les
surfaces urbaines restituent la chaleur accumulée, et c’est généralement à
ce moment-là que le contraste urbain/rural atteint son maximum.

Sur la carte moyenne, le cœur de l’îlot reste bien fixé sur l’hypercentre,
en ligne avec les résultats présentés dans l’article. On observe également
une extension vers le sud, un motif qui reflète assez clairement la
distribution des stations: contrairement à ce qui se passait en
journée, ce n’est pas un point isolé qui oriente l’interpolation, mais
plusieurs stations qui tirent la structure dans cette direction. Avec un
réseau plus dense, on verrait sans doute une extension un peu plus
affirmée vers le nord aussi, ce que la littérature suggère déjà.

L’organisation générale de la carte est donc cohérente avec ce qu’on
attend d’un îlot de chaleur nocturne marqué. Les détails fins restent à
considérer avec prudence, mais l’équilibre global apparaît solide compte
tenu du nombre de stations disponibles.
"""
)

night_title = (
    f"ICU moyen – NUIT (heures {min(night_hours)}–{max(night_hours)}h, "
    f"{len(hottest_days)} jours les plus chauds)"
)
plot_cumulative_overlay(
    night_cumulative_mean,
    grid_lon,
    grid_lat,
    night_title,
    urban_stations_df,
    urban_center,
)

# --------------------------------------------------------------------
# Conclusion markdown (texte inchangé)
# --------------------------------------------------------------------
st.markdown(
    """
**Conclusion**

Ce prototype propose une première visualisation des îlots de chaleur à
Rennes durant l’été 2018, en se concentrant sur les journées les plus
chaudes de la période juin–août. Les cartes obtenues permettent déjà
d’illustrer simplement le phénomène : elles mettent en évidence les
contrastes jour/nuit, révèlent des points chauds récurrents et offrent un
aperçu visuel clair, utile pour quelqu’un qui découvre le sujet. Les
résultats vont d’ailleurs globalement dans le même sens que ceux décrits
dans la littérature, ce qui est encourageant.

Dans ce travail, je n’ai utilisé que les températures horaires issues des
23 stations disponibles en 2018. C’est un sous-ensemble du réseau RUN, qui
aujourd’hui combine plusieurs types de capteurs : des stations automatiques
DAVIS mesurant au minimum la température, l’humidité relative, le vent, la
pluie et la pression (parfois le rayonnement), et un réseau plus dense de
capteurs LoRaWAN enregistrant température et humidité. Toutes les stations
ne mesurent pas l’ensemble de ces variables, mais le réseau fournit tout de
même une base beaucoup plus riche que celle exploitée ici.

Une suite naturelle consisterait donc à mobiliser ces données
complémentaires pour analyser des combinaisons de facteurs plus fines :
par exemple repérer les situations les plus défavorables (peu de vent,
ciel clair, faible humidité, végétation limitée), ou comprendre comment
les caractéristiques du bâti et de la végétation structurent réellement
l’ICU. On pourrait aussi croiser ces cartes avec d’autres sources : champs
météorologiques plus larges, imagerie satellitaire (indices de végétation,
thermique), ou classifications du bâti et de la végétation issues des
données IGN.

À mesure que l’on combine ces informations, des approches statistiques, de
machine learning ou même de deep learning pourraient devenir pertinentes —
non pas pour remplacer le travail des urbanistes, mais pour leur fournir
des outils plus riches. L’idée serait qu’ils puissent interagir eux-mêmes
avec les données, explorer différents réglages, lancer leurs propres
analyses et bénéficier d’indicateurs plus précis pour appuyer leurs choix.
À plus long terme, on peut même imaginer un volet de modélisation, où ils
pourraient tester différents scénarios d’aménagement ou d’adaptation pour
comparer leurs effets sur l’îlot de chaleur.

L’enjeu final est de contribuer à rendre la ville plus vivable lors des
épisodes de chaleur estivale, en particulier pour les populations les plus
vulnérables. On sait que les canicules s’accompagnent d’une surmortalité
marquée, notamment chez les personnes âgées : disposer d’outils plus
précis, interactifs et adaptés aux besoins des professionnels pourrait
participer à limiter ces impacts.

Au-delà de l’aspect opérationnel, une démarche de ce type pourrait aussi
s’inscrire dans un effort de recherche plus large. En croisant des données
locales denses, des informations géographiques détaillées et des méthodes
d’analyse plus avancées, on pourrait contribuer à améliorer la
compréhension des îlots de chaleur urbains et, plus largement, des
questions d’adaptation au changement climatique. Les outils développés
pourraient être réutilisés ou adaptés par d’autres collectivités, ou encore
servir de base à des travaux menés dans d’autres régions ou d’autres pays,
ce qui favoriserait une science de l’ICU plus ouverte et cumulative.

Dans un contexte où les populations se concentrent de plus en plus en ville
et où les épisodes de chaleur s’intensifient, disposer de méthodes
robustes, partageables et réellement utiles serait une avancée importante.
Sans prétendre répondre à tous les enjeux, ce type d’approche pourrait
participer, à son échelle, à l’amélioration des connaissances et à la
conception d’outils d’aide à la décision plus adaptés aux défis à venir.
"""
)
