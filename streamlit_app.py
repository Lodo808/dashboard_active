import streamlit as st
import pandas as pd

import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium

from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta
import math
from openai import OpenAI

st.set_page_config(page_title="Dashboard Activelabel", layout="wide")
st.markdown("""
    <style>
        /* Sfondo principale e contenitori */
        body, .stApp {
            background-color: white !important;
            color: black !important;
        }

        /* Testo e componenti principali */
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: white !important;
        }

        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }

        /* Tabelle e widget */
        div[data-testid="stDataFrame"] {
            background-color: white !important;
        }
        div[data-testid="stDataFrame"] * {
            color: black !important;
        }

        /* Metriche e card */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            color: #2E4053 !important;
        }

        /* Rimuove eventuale overlay scuro del tema */
        [class*="st-emotion-cache"] {
            background-color: white !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=250)
with col2:
    st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; padding-top: 45px;'>
            <h1 style='font-size: 60px; margin: 0;'>Dashboard Activelabel</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
div[data-testid="stDataFrame"] table {
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
thead tr th {
    background-color: #f4f6f8 !important;
    color: #2E4053 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# Link al file Drive (pubblico, leggibile)
file_id = "1Zp_2qP8Td1TVMdbY9mlzOHYxIlICmNZL"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

geolocator = Nominatim(user_agent="dashboard_active_label")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_province(lat, lon):
    try:
        location = reverse((lat, lon), language="it")
        if location and "address" in location.raw:
            address = location.raw["address"]
            # 🔍 Proviamo diversi campi in ordine di importanza
            return (
                address.get("state_district") or
                address.get("county") or
                address.get("state") or
                address.get("region") or
                address.get("city") or
                address.get("town") or
                address.get("village") or
                address.get("municipality") or
                address.get("country")
            )
    except Exception:
        return None
    return None

try:
    # Caricamento del file
    df = pd.read_csv(url, sep=" ", header=None)
    df.columns = ["Operator", "Device", "Reading date", "Reading hour", "Read value", "Effective T", "QR", "Desired T",
                  "Item ID", "Writing date", "Writing hour", "LAT", "LON"]

    @st.cache_data(show_spinner="Determinazione province in corso...")
    def get_province_cached(lat, lon):
        try:
            location = reverse((lat, lon), language="it")
            if location and "address" in location.raw:
                address = location.raw["address"]
                # 🔍 Proviamo diversi campi in ordine di importanza
                return (
                        address.get("state_district") or
                        address.get("county") or
                        address.get("state") or
                        address.get("region") or
                        address.get("city") or
                        address.get("town") or
                        address.get("village") or
                        address.get("municipality") or
                        address.get("country")
                )
        except Exception:
            return None
        return None

    # Applichiamo la funzione solo se LAT e LON sono validi
    df["Province"] = df.apply(
        lambda r: get_province_cached(r["LAT"], r["LON"]) if pd.notna(r["LAT"]) and pd.notna(r["LON"]) else None,
        axis=1
    )

    mappa_prodotti = {
        "B": {"nome": "Banana", "scadenza_giorni": 7},
        "M": {"nome": "Mela", "scadenza_giorni": 10},
        "A": {"nome": "Arancia", "scadenza_giorni": 14},
        "L": {"nome": "Latte", "scadenza_giorni": 5},
        "Y": {"nome": "Yogurt", "scadenza_giorni": 12},
        "P": {"nome": "Parmigiano", "scadenza_giorni": 24},
    }
    scadenze_per_prodotto = {v["nome"]: v["scadenza_giorni"] for k, v in mappa_prodotti.items()}

    def assegna_prodotto_e_scadenza(df):
        df = df.copy()
        df["Reading date"] = pd.to_datetime(df["Reading date"], dayfirst=True, errors="coerce")

        prodotti, scadenze_iniziali, scadenze_residue = [], [], []
        prima_lettura = {}

        for _, row in df.iterrows():
            qr_val = str(row.get("QR", "")).strip()
            qr_val = qr_val.replace(" ", "").upper()

            if not qr_val or qr_val == "NAN":
                prodotti.append(None)
                scadenze_iniziali.append(None)
                scadenze_residue.append(None)
                continue

            iniziale = qr_val[0]
            prodotto_info = mappa_prodotti.get(iniziale, {"nome": "Sconosciuto", "scadenza_giorni": 7})
            nome_prodotto = prodotto_info["nome"]
            giorni_scadenza = prodotto_info["scadenza_giorni"]

            data_lettura = row["Reading date"]
            if pd.isna(data_lettura):
                prodotti.append(nome_prodotto)
                scadenze_iniziali.append(None)
                scadenze_residue.append(None)
                continue

            if qr_val not in prima_lettura:
                prima_lettura[qr_val] = data_lettura
                data_scadenza = data_lettura + timedelta(days=giorni_scadenza)
                scadenza_residua = giorni_scadenza
            else:
                giorni_passati = (data_lettura - prima_lettura[qr_val]).days
                scadenza_residua = giorni_scadenza - giorni_passati
                data_scadenza = prima_lettura[qr_val] + timedelta(days=giorni_scadenza)

            prodotti.append(nome_prodotto)
            scadenze_iniziali.append(data_scadenza.date())
            scadenze_residue.append(scadenza_residua)

        df["Product"] = prodotti
        df["Expiry date (initial)"] = scadenze_iniziali
        df["Reading date"] = df["Reading date"].dt.strftime("%d/%m/%Y")
        df["Expiry date (initial)"] = pd.to_datetime(df["Expiry date (initial)"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
        df["Days left"] = scadenze_residue
        return df

    df = assegna_prodotto_e_scadenza(df)
    with st.spinner("⏳ Calcolo colonne aggiuntive in corso..."):
        df = assegna_prodotto_e_scadenza(df)
        df["Province"] = df.apply(
            lambda r: get_province_cached(r["LAT"], r["LON"]) if pd.notna(r["LAT"]) and pd.notna(r["LON"]) else None,
            axis=1
        )


    def _expiry_factor_from_days(g, initial_shelf_life):
        if pd.isna(g):
            return 0.0
        if initial_shelf_life == 0:
            return 0.0 if g < 0 else 1.0
        g = float(g)
        if g >= 0:
            return 0.6 + 0.4 * (g / initial_shelf_life)
        elif g >= -7:
            return 0.3 + (0.6 - 0.3) * ((g + 7.0) / 7.0)
        elif g >= -14:
            return 0.25 + (0.3 - 0.25) * ((g + 14.0) / 7.0)

        return max(0.0, 0.25 * math.exp((g + 14.0) / 14.0))


    def calcola_indice_freschezza(row):
        temp_diff = abs(row["Effective T"] - row["Desired T"])
        temp_factor = max(0.0, 1.0 - (temp_diff / 10.0))
        giorni_residui = row.get("Days left", 0)
        nome_prodotto = row.get("Product", "Sconosciuto")
        initial_shelf_life = scadenze_per_prodotto.get(nome_prodotto, 7)
        expiry_factor = _expiry_factor_from_days(giorni_residui, initial_shelf_life)
        combined = (temp_factor ** 0.6) * (expiry_factor ** 1.0)
        return round(100.0 * combined, 1)


    def calcola_waste_cost(df):
        prezzi_medi = {
            "Banana": 0.3,
            "Mela": 0.5,
            "Arancia": 0.6,
            "Latte": 1.2,
            "Yogurt": 1.0,
            "Sconosciuto": 0.5
        }

        prodotti_sprecati = df[df["freshness_index"] < 50]["Product"]
        costo_totale = sum(prezzi_medi.get(p, 0.5) for p in prodotti_sprecati)
        return round(costo_totale, 2)

    # Calcolo dell'indice di freschezza se non esiste ancora
    if "freshness_index" not in df.columns:
        df["freshness_index"] = df.apply(calcola_indice_freschezza, axis=1)

    tipo = "Car"
    fattori = {"Car": 0.12, "Truck": 0.6, "Refrigerated Truck": 0.9}
    fattore_emissione = fattori[tipo]

    # --- Calcolo metriche generali su tutti i dati ---
    total_shipments = len(df)
    compliant = len(df[df["freshness_index"] >= 50])
    incidenti = len(df[df["freshness_index"] < 50])
    waste_cost = calcola_waste_cost(df)  # esempio: 10€ per prodotto "non conforme"

    perc_compliant = (compliant / total_shipments * 100) if total_shipments > 0 else 0
    perc_incident = (incidenti / total_shipments * 100) if total_shipments > 0 else 0

    st.markdown("""
        <style>
            .snapshot-title {
                font-family: 'Segoe UI', sans-serif;
                font-size: 42px;
                font-weight: 700;
                text-align: center;
                color: #2E4053;
                letter-spacing: 1px;
                margin-top: 10px;
                margin-bottom: 30px;
            }
            div[data-testid="stMetric"] {
                background-color: #f8f9fa;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                text-align: center;
                transition: all 0.2s ease-in-out;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            @media (max-width: 900px) {
                div[data-testid="column"] {
                    flex: 1 1 50% !important;
                    min-width: 240px !important;
                }
            }
            @media (max-width: 600px) {
                div[data-testid="column"] {
                    flex: 1 1 100% !important;
                }
            }
        </style>
        <div class="snapshot-title">🚦 Executive Snapshot</div>
    """, unsafe_allow_html=True)

    # --- Layout colonne metriche ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ % Compliant Shipments", f"{perc_compliant:.1f}%")
    with col2:
        st.metric("⚠️ % Shipments with Incidents", f"{perc_incident:.1f}%")
    with col3:
        st.metric("📦 Total Shipments", total_shipments)
    with col4:
        st.metric("💸 Total Waste Cost (€)", f"{waste_cost:,.2f}")

    st.markdown("---")

    operatori_disponibili = sorted(df["Operator"].dropna().unique().tolist())

    operatori_selezionati = st.multiselect(
        "Filter by operator",
        operatori_disponibili,
        help="Select one or more operators to filter the main table"
    )
    if operatori_selezionati:
        filtered = df[df["Operator"].isin(operatori_selezionati)]
    else:
        filtered = df

    filtered = filtered.copy()

    filtered["Reading date"] = pd.to_datetime(filtered["Reading date"], dayfirst=True, errors="coerce")
    filtered["Expiry date (initial)"] = pd.to_datetime(filtered["Expiry date (initial)"], dayfirst=True, errors="coerce")
    filtered["Writing date"] = pd.to_datetime(filtered["Writing date"], dayfirst=True, errors="coerce")

    colonne_da_mostrare = [
        "Operator", "Reading date", "Read value", "Effective T", "QR",
        "Desired T", "Product", "freshness_index"
    ]

    if 'selected_qr' not in st.session_state:
        st.session_state.selected_qr = "All"

    if 'df_selection' not in st.session_state:
        st.session_state.df_selection = []

    st.write("Table of recent scans. Click on a row to see the history and map of that QR code.")

    filtered_last_scan = filtered.dropna(subset=["Reading date"]) \
        .sort_values("Reading date") \
        .groupby("QR") \
        .tail(1)

    df_sorted = filtered_last_scan.sort_values(
        "Reading date", ascending=False
    ).reset_index(drop=True).copy()

    df_to_display = df_sorted[colonne_da_mostrare].copy()

    event = st.dataframe(
        df_to_display,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Reading date": st.column_config.DatetimeColumn(
                "Reading date",
                format="DD/MM/YYYY",
            )
        }
    )

    new_selection = event.selection.rows
    old_selection = st.session_state.df_selection

    if new_selection:
        selected_row_index = new_selection[0]
        qr_from_click = df_sorted.iloc[selected_row_index]["QR"]
        st.session_state.selected_qr = qr_from_click

    elif not new_selection and old_selection:
        st.session_state.selected_qr = "All"
    st.session_state.df_selection = new_selection

    st.subheader("📜 Scan history and details")
    unique_qr = filtered["QR"].dropna().unique().tolist()

    options = ["All"] + unique_qr
    if st.session_state.selected_qr not in options:
        st.session_state.selected_qr = "All"

    selected_qr = st.selectbox(
        "Select QR to view only its scans (or click a row in the table)",
        options,
        key="selected_qr"
    )

    if selected_qr != "All":
        map_data = filtered[filtered["QR"] == selected_qr]

        # --- 📜 Storico delle scansioni ---
        storico = map_data.sort_values("Reading date", ascending=True)

        st.dataframe(
            storico,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Reading date": st.column_config.DatetimeColumn(
                    "Reading date",
                    format="DD/MM/YYYY",
                ),
                "Writing date": st.column_config.DatetimeColumn(
                    "Writing date",
                    format="DD/MM/YYYY",
                ),
                "Expiry date (initial)": st.column_config.DatetimeColumn(
                    "Expiry date (initial)",
                    format="DD/MM/YYYY",
                )
            }
        )
        map_zoom = 6
        map_data_for_plot = storico
    else:
        map_data = filtered
        map_zoom = 4

    # --- Mappa con Plotly ---
    st.subheader("🗺️ Geolocation of QR Scans")

    if not map_data.empty and "LAT" in map_data.columns and "LON" in map_data.columns:

        map_data_valid = map_data.dropna(subset=["LAT", "LON"])

        if map_data_valid.empty:
            st.warning("Nessun dato GPS valido da mostrare sulla mappa.")
        else:
            center_lat = map_data_valid["LAT"].mean()
            center_lon = map_data_valid["LON"].mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=map_zoom,
                tiles="cartodbdarkmatter"
            )

            def get_color(freshness):
                if pd.isna(freshness): return "grey"
                if freshness >= 80: return "green"
                if freshness >= 50: return "orange"
                return "red"

            for _, row in map_data_valid.iterrows():
                tooltip_html = f"""
                        <b>QR:</b> {row['QR']}<br>
                        <b>Prodotto:</b> {row['Product']}<br>
                        <b>Freschezza:</b> {row['freshness_index']}%<br>
                        <b>Data:</b> {row['Reading date'].strftime('%d/%m/%Y')}<br>
                        <b>Ora:</b> {row['Reading hour']}
                    """

                folium.CircleMarker(
                    location=(row["LAT"], row["LON"]),
                    radius=7,
                    color=get_color(row["freshness_index"]),
                    fill=True,
                    fill_color=get_color(row["freshness_index"]),
                    fill_opacity=0.7,
                    tooltip=folium.Tooltip(tooltip_html)
                ).add_to(m)

            if selected_qr != "All":
                line_data = map_data_for_plot.dropna(subset=["LAT", "LON"])

                if len(line_data) > 1:
                    path_coords = list(zip(line_data["LAT"], line_data["LON"]))

                    AntPath(
                        locations=path_coords,
                        use_hardware_acceleration=True,
                        delay=1000,
                        dash_array=[10, 15],
                        weight=5,
                        color="#00AEEF",
                        pulse_color="#FFFFFF",
                        reverse=False,
                    ).add_to(m)

                    min_lat = line_data["LAT"].min()
                    max_lat = line_data["LAT"].max()
                    min_lon = line_data["LON"].min()
                    max_lon = line_data["LON"].max()

                    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
                    m.fit_bounds(bounds, padding=(10, 10))

            st.markdown("""
                <style>
                    .leaflet-control-attribution {
                        display: none !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            st_folium(
                m,
                use_container_width=True,
                height=550,
                returned_objects=[]
            )

except Exception as e:
    st.error(f"Errore nel caricamento del file: {e}")

# --- Riepilogo freschezza per QR (solo temperature) ---
st.markdown("---")
st.subheader("❄️ QR Freshness Summary")

df["Reading date"] = pd.to_datetime(df["Reading date"], dayfirst=True, errors="coerce")

df["Expiry date (initial)"] = pd.to_datetime(df["Expiry date (initial)"], dayfirst=True, errors="coerce")
df["Writing date"] = pd.to_datetime(df["Writing date"], dayfirst=True, errors="coerce")

first_scans = df.loc[df.groupby("QR")["Reading date"].idxmin()][["QR", "Reading date"]]
first_scans = first_scans.rename(columns={"Reading date": "First Scan Date"})

last_scans = (
    df.sort_values("Reading date")
      .groupby("QR")
      .tail(1)
      .copy()
)

last_scans = pd.merge(last_scans, first_scans, on="QR", how="left")
last_scans["freshness_index"] = last_scans.apply(calcola_indice_freschezza, axis=1)

# Aggiungiamo una valutazione qualitativa
def freshness_status(value):
    if value >= 80:
        return "🟢 Excellent"
    elif value >= 50:
        return "🟡 Moderate"
    else:
        return "🔴 Poor"

last_scans["Status"] = last_scans["freshness_index"].apply(freshness_status)

if selected_qr != "All":
    summary_data = last_scans[last_scans["QR"] == selected_qr]
    st.caption(f"Showing latest freshness data for QR **{selected_qr}**")
else:
    summary_data = last_scans
    st.caption("Showing latest freshness data for all QR codes")

cols = ["QR", "First Scan Date", "Expiry date (initial)", "Days left", "Desired T", "Effective T", "freshness_index", "Status"]
summary_data_display = summary_data[cols].rename(columns={
    "First Scan Date": "First Scan Date",
    "Expiry date (initial)": "Expiry Date",
    "Days left": "Days Left"
})

st.dataframe(
    summary_data[cols].sort_values("freshness_index"),
    use_container_width=True,
    hide_index=True,
    column_config={
        "First Scan Date": st.column_config.DatetimeColumn(
            "First Scan Date",
            format="DD/MM/YYYY",
        ),
        "Expiry date (initial)": st.column_config.DatetimeColumn(
            "Expiry date (initial)",
            format="DD/MM/YYYY",
        )
    }
)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Raggio terrestre in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# --- Calcolo distanza per ogni veicolo e ripartizione CO₂ tra i QR ---
distances = []
for operator, group_op in df.groupby("Operator"):
    group_op = group_op.sort_values("Reading date")
    total_dist = 0.0
    coords = list(zip(group_op["LAT"], group_op["LON"]))

    # Calcolo distanza totale percorsa dall'operatore
    for i in range(1, len(coords)):
        if all(not pd.isna(x) for x in (*coords[i-1], *coords[i])):
            total_dist += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])

    # --- Emissioni totali dell'operatore ---
    qrs = group_op["QR"].dropna().unique()
    distances.append({
        "Operator": operator,
        "Distance_km": total_dist,
        "QR_list": qrs
    })

df_distance = pd.DataFrame(distances)

# --- Selettore tipo di trasporto ---
st.markdown("---")
st.subheader("🌱 CO₂ Emissions by QR (based on travelled distance)")

col1, col2, col3 = st.columns([1.5, 1.5, 1])

operatori = sorted(df_distance["Operator"].dropna().unique().tolist())
selected_operator = col1.multiselect("👷 Operator", operatori, help="Select one or more operators")
if selected_operator:
    qrs_filtrabili = sorted([
        qr
        for op in selected_operator
        for qr_list in df_distance.loc[df_distance["Operator"] == op, "QR_list"]
        for qr in qr_list
    ])
else:
    # Nessun operatore selezionato → mostra tutti i QR
    qrs_filtrabili = sorted(df["QR"].dropna().unique().tolist())

selected_qr_filter = col2.multiselect("🔢 QR Code", qrs_filtrabili, help="Filter by QR code")

# 3️⃣ Filtro per intervallo di distanza
min_dist, max_dist = float(df_distance["Distance_km"].min()), float(df_distance["Distance_km"].max())

if min_dist < max_dist:
    selected_distance = col3.slider("🚚 Distance (km)", min_dist, max_dist, (min_dist, max_dist))
else:
    col3.info("Filtro distanza non disponibile (dati insufficienti).")
    selected_distance = (min_dist, max_dist)

tipo = st.selectbox("Transport type", ["Car", "Truck", "Refrigerated Truck"])
fattori = {"Car": 0.12, "Truck": 0.6, "Refrigerated Truck": 0.9}
fattore_emissione = fattori[tipo]

# --- Calcolo emissioni CO₂ e ripartizione tra QR ---
expanded_rows = []
for _, row in df_distance.iterrows():
    emissioni_totali = row["Distance_km"] * fattore_emissione
    qrs = row["QR_list"]
    if len(qrs) > 0:
        emissioni_per_qr = emissioni_totali / len(qrs)
        for qr in qrs:
            expanded_rows.append({
                "Operator": row["Operator"],
                "QR": qr,
                "Distance_km": row["Distance_km"],
                "Emissions_CO2_kg": emissioni_per_qr
            })

df_emissioni = pd.DataFrame(expanded_rows)

df_emissioni_filtered = df_emissioni.copy()
if selected_operator:
    df_emissioni_filtered = df_emissioni_filtered[df_emissioni_filtered["Operator"].isin(selected_operator)]
if selected_qr_filter:
    df_emissioni_filtered = df_emissioni_filtered[df_emissioni_filtered["QR"].isin(selected_qr_filter)]
df_emissioni_filtered = df_emissioni_filtered[
    df_emissioni_filtered["Distance_km"].between(*selected_distance)
]

# --- Visualizzazione tabella ---
st.dataframe(
    df_emissioni_filtered.sort_values("Emissions_CO2_kg", ascending=False),
    use_container_width=True,
    hide_index=True
)

# --- Metrica totale ---
totale_co2 = df_emissioni_filtered["Emissions_CO2_kg"].sum()
st.metric("Total estimated CO₂", f"{totale_co2:.2f} kg")

# --- Sezione finale: AI Analyst ---
ai_container = st.container()

with ai_container:
    st.markdown("---")
    st.subheader("🤖 AI Analyst")

    st.markdown(
        "Click one of the buttons below to automatically generate a report"
        "based on the processed data (freshness, emissions, shipments, operators)."
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ No API Key configured. Add it to Streamlit Secrets or directly in the code.")
    else:
        client = OpenAI(api_key=api_key)
        if "report_ai" not in st.session_state:
            st.session_state.report_ai = None
        if "report_lang" not in st.session_state:
            st.session_state.report_lang = None

        # --- Pulsanti per la lingua ---
        col1, col2, col3 = st.columns([1.5, 1.5, 3])
        with col1:
            genera_report_it = st.button("📊 Genera Report (IT)")
        with col2:
            genera_report_en = st.button("📊 Generate Report (EN)")

        summary = {
            "Total shipments": len(df),
            "% compliant": round(perc_compliant, 2),
            "% incidents": round(perc_incident, 2),
            "Waste cost (€)": waste_cost,
            "Total estimated CO2 (kg)": round(totale_co2, 2),
            "Average freshness index": round(df["freshness_index"].mean(), 2),
            "Unique operators": df["Operator"].nunique(),
            "Products analyzed": df["Product"].nunique(),
        }

        # --- Definizioni dei Prompt e testi ---
        prompts = {
            'it': {
                'spinner': "Analisi in corso...",
                'system': "Sei un analista esperto di logistica e qualità del trasporto alimentare.",
                'user': f"""Sei un data analyst. Analizza i dati seguenti e genera un report completo.
                         Rispondi in modo chiaro e strutturato, **IN ITALIANO**, suddividendo in:
                         - Panoramica generale
                         - Punti critici o anomalie
                         - Raccomandazioni operative

                         Dati sintetici: {summary}""",
                'success': "✅ Analisi completata (IT)",
                'follow_up_header': "### 💬 Fai altre domande sui dati (IT)",
                'follow_up_input': "Scrivi una domanda",
                'spinner_question': "Elaborazione risposta...",
                'response_header': "### 🔍 Risposta AI",
                'context_system': "Sei un assistente analitico che risponde **IN ITALIANO** su dati di logistica e qualità alimentare."
            },
            'en': {
                'spinner': "Analysis in progress...",
                'system': "You are an expert analyst in logistics and food transport quality.",
                'user': f"""You are a data analyst. Analyze the following data and generate a complete report.
                         Respond clearly and structured, **IN ENGLISH**, divided into:
                         - General overview
                         - Critical points or anomalies
                         - Operational recommendations

                         Synthetic data: {summary}""",
                'success': "✅ Analysis completed (EN)",
                'follow_up_header': "### 💬 Ask more about the data (EN)",
                'follow_up_input': "Write a question",
                'spinner_question': "Processing response...",
                'response_header': "### 🔍 AI Response",
                'context_system': "You are an analytical assistant answering **IN ENGLISH** about logistics and food quality data."
            }
        }

        lang_to_generate = None
        if genera_report_it:
            lang_to_generate = 'it'
        elif genera_report_en:
            lang_to_generate = 'en'

        # --- Generazione Report ---
        if lang_to_generate:
            T = prompts[lang_to_generate]
            with st.spinner(T['spinner']):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": T['system']},
                        {"role": "user", "content": T['user']}
                    ],
                    temperature=0.5,
                )

                st.session_state.report_ai = response.choices[0].message.content
                st.session_state.report_lang = lang_to_generate

        if st.session_state.report_ai:

            T = prompts.get(st.session_state.report_lang, prompts['en'])

            st.success(T['success'])
            st.markdown("### 📈 AI Report")
            st.write(st.session_state.report_ai)

            # --- Sezione Domande Aggiuntive ---
            st.markdown(T['follow_up_header'])
            user_question = st.text_input(
                T['follow_up_input'],
                key=f"q_input_{st.session_state.report_lang}"
            )

            if user_question:
                with st.spinner(T['spinner_question']):
                    context = df.describe(include="all").to_string()
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": T['context_system']},
                            {"role": "user", "content": f"Domanda: {user_question}\n\nContesto dati:\n{context}"}
                        ],
                        temperature=0.4,
                    )
                    st.markdown(T['response_header'])
                    st.write(response.choices[0].message.content)