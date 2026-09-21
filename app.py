import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import unicodedata
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Sequía IIPAC", layout="wide")

try:
    st.sidebar.image("LogoIIPAC.jpg", use_container_width=True)
except:
    st.sidebar.warning("Logo no encontrado. Subí 'LogoIIPAC.jpg'.")

st.title("Análisis de la Sequía 2020-2023 (Evento Niña)")
st.markdown(
    """
    <div style="font-size: 17px; line-height: 1.6; color: #444; margin-bottom: 1.5rem;">
        <b>Análisis de la intensidad y duración de la sequía en relación a las resoluciones de emergencia declaradas.</b><br> 
        El valor de sequía corresponde al valor acumulado trimestral, sumatoria de los valores de intensidad de sequía: leve = 1, moderada = 2 y severa = 3.<br>
        Se representan dos curvas: el valor <b>mediana</b> y el valor <b>máximo</b> departamental.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────
def normalizar_nombre(nombre):
    nombre = str(nombre).strip().lower()
    nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre)
                     if unicodedata.category(c) != 'Mn')
    for prefijo in ['partido de ', 'partido del ', 'departamento de ',
                    'departamento del ', 'dpto. ', 'dpto ', 'partido ']:
        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):]
    return re.sub(r'[^a-z0-9]', '', nombre)

def buscar_columna(df, patrones):
    cols_norm = {normalizar_nombre(c): c for c in df.columns}
    for p in patrones:
        pn = normalizar_nombre(p)
        for cn, cr in cols_norm.items():
            if pn in cn or cn in pn:
                return cr
    return None

def match_departamento(depto_seq, deptos_eme_norm):
    d_seq_norm = normalizar_nombre(depto_seq)
    if not d_seq_norm:
        return None
    for d_eme_orig, d_eme_norm in deptos_eme_norm.items():
        if d_seq_norm == d_eme_norm:
            return d_eme_orig
    for d_eme_orig, d_eme_norm in deptos_eme_norm.items():
        if d_seq_norm in d_eme_norm or d_eme_norm in d_seq_norm:
            return d_eme_orig
    return None

# ─────────────────────────────────────────────────────────────────────
# COLORES POR ACTIVIDAD
# ─────────────────────────────────────────────────────────────────────
COLORES_ACTIVIDAD = {
    'agriculturafamiliar': '#808080',
    'agricultura': '#90EE90',
    'ganaderia': '#8B4513',
    'apicultura': '#FFD700',
    'fruticultura': '#FF0000',
    'horticultura': '#FFA500',
    'psicultura': '#87CEEB',
    'piscicultura': '#87CEEB',
    'silvicultura': '#006400',
}

def color_para_actividad(act):
    act_norm = normalizar_nombre(act)
    for key, color in COLORES_ACTIVIDAD.items():
        if key in act_norm:
            return color
    return '#cccccc'

# ─────────────────────────────────────────────────────────────────────
# COLORES POR CLUSTER (para el mapa)
# ─────────────────────────────────────────────────────────────────────
COLORES_CLUSTER = {
    0: '#e41a1c',  # rojo
    1: '#377eb8',  # azul
    2: '#4daf4a',  # verde
    3: '#984ea3',  # violeta
    4: '#ff7f00',  # naranja
    5: '#a65628',  # marrón
    6: '#f781bf',  # rosa
    7: '#999999',  # gris
}

# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    URL_MED = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_sequia_mediana_v2.csv"
    URL_MAX = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_sequia_maximo.csv"
    URL_EME = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_todo_V2.csv"

    def leer_base_sequia(url, sufijo):
        df = pd.read_csv(url, sep=',', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        cols = [c for c in df.columns if c.endswith(sufijo)]
        mapa = {}
        for c in cols:
            m = re.match(r'([A-Z]{3})(\d{4})' + sufijo, c)
            if m:
                mapa[c] = f"{m.group(1)}_{m.group(2)}"
        df = df.rename(columns=mapa)
        periodos = list(mapa.values())
        for p in periodos:
            df[p] = pd.to_numeric(df[p], errors='coerce')
        return df, periodos

    # ── Base de sequía MEDIANA ──────────────────────────────────────
    df_med, periodos_med = leer_base_sequia(URL_MED, '_median')
    col_prov_med = buscar_columna(df_med, ['provincia'])
    col_dept_med = buscar_columna(df_med, ['departamento', 'nam', 'partido', 'municipio'])
    if not col_prov_med or not col_dept_med:
        st.error("❌ No se detectaron columnas de PROVINCIA/DEPARTAMENTO en la base de mediana.")
        st.stop()
    df_med['PROVINCIA']    = df_med[col_prov_med].astype(str).str.strip()
    df_med['DEPARTAMENTO'] = df_med[col_dept_med].astype(str).str.strip()
    df_med['_prov_norm']   = df_med['PROVINCIA'].apply(normalizar_nombre)
    df_med['_dept_norm']   = df_med['DEPARTAMENTO'].apply(normalizar_nombre)

    # ── Base de sequía MÁXIMO ──────────────────────────────────────
    df_max, periodos_max = leer_base_sequia(URL_MAX, '_max')
    col_prov_max = buscar_columna(df_max, ['provincia'])
    col_dept_max = buscar_columna(df_max, ['departamento', 'nam', 'partido', 'municipio'])
    if not col_prov_max or not col_dept_max:
        st.error("❌ No se detectaron columnas de PROVINCIA/DEPARTAMENTO en la base de máximo.")
        st.stop()
    df_max['PROVINCIA']    = df_max[col_prov_max].astype(str).str.strip()
    df_max['DEPARTAMENTO'] = df_max[col_dept_max].astype(str).str.strip()
    df_max['_prov_norm']   = df_max['PROVINCIA'].apply(normalizar_nombre)
    df_max['_dept_norm']   = df_max['DEPARTAMENTO'].apply(normalizar_nombre)

    # ── Base de emergencia ──────────────────────────────────────────
    df_eme = pd.read_csv(URL_EME, sep=',', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
    df_eme.columns = df_eme.columns.str.strip()
    cols_eme = [c for c in df_eme.columns if re.match(r'^[A-Z]{3}_\d{4}$', c)]
    periodos_eme = list(cols_eme)
    col_prov_eme = buscar_columna(df_eme, ['provincia'])
    col_dept_eme = buscar_columna(df_eme, ['departamento', 'nam', 'partido', 'municipio'])
    col_act_eme  = buscar_columna(df_eme, ['actividad'])
    if not col_prov_eme or not col_dept_eme or not col_act_eme:
        st.error("❌ No se detectaron todas las columnas necesarias en BBDD_todo_V2.")
        st.stop()
    df_eme['PROVINCIA']    = df_eme[col_prov_eme].astype(str).str.strip()
    df_eme['DEPARTAMENTO'] = df_eme[col_dept_eme].astype(str).str.strip()
    df_eme['ACTIVIDAD']    = df_eme[col_act_eme].astype(str).str.strip()
    df_eme['_prov_norm']   = df_eme['PROVINCIA'].apply(normalizar_nombre)
    df_eme['_dept_norm']   = df_eme['DEPARTAMENTO'].apply(normalizar_nombre)

    # Unión de períodos (mediana + emergencia)
    periodos_todos = list(periodos_med)
    for p in periodos_eme:
        if p not in periodos_todos:
            periodos_todos.append(p)

    return df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos

df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos = load_data()

# ─────────────────────────────────────────────────────────────────────
# CLUSTERING FIJO (K-MEANS sobre la mediana)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def calcular_clusters(df_med, periodos_med, k=5):
    """Calcula clustering K-means sobre las series de mediana, con k fijo."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Matriz de series (solo filas sin NaN)
    df_valid = df_med.dropna(subset=periodos_med).copy()
    if df_valid.empty:
        return df_med.assign(cluster=-1)

    X = df_valid[periodos_med].values
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df_valid['cluster'] = km.fit_predict(X_z)

    # Devolver todo el df_med con la columna cluster
    df_out = df_med.merge(
        df_valid[['PROVINCIA', 'DEPARTAMENTO', 'cluster']],
        on=['PROVINCIA', 'DEPARTAMENTO'],
        how='left'
    )
    df_out['cluster'] = df_out['cluster'].fillna(-1).astype(int)
    return df_out

df_med = calcular_clusters(df_med, periodos_med, k=5)

# ─────────────────────────────────────────────────────────────────────
# CARGA DEL GEOPACKAGE
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_geopackage():
    """Carga el GeoPackage de departamentos del área de estudio."""
    import geopandas as gpd

    # Opción A: archivo local en el repo
    try:
        gdf = gpd.read_file("Departamentos_area_estudio.gpkg")
    except Exception:
        # Opción B: descargar desde GitHub raw
        url = "https://raw.githubusercontent.com/emiliano2026/sequia/main/Departamentos_area_estudio.gpkg"
        gdf = gpd.read_file(url)

    # Asegurar CRS WGS84 (lat/lon)
    if gdf.crs is not None and gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    return gdf

try:
    gdf_deptos = cargar_geopackage()
    GEOPACKAGE_OK = True
except Exception as e:
    st.sidebar.warning(f"⚠️ No se pudo cargar el GeoPackage: {e}")
    gdf_deptos = None
    GEOPACKAGE_OK = False

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "depto_sel" not in st.session_state:
    st.session_state["depto_sel"] = None
if "prov_sel" not in st.session_state:
    st.session_state["prov_sel"] = None

# ─────────────────────────────────────────────────────────────────────
# MAPA INTERACTIVO
# ─────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Mapa interactivo — clic en un departamento para filtrar")

if GEOPACKAGE_OK and gdf_deptos is not None:
    # Detectar columna de nombre en el GeoPackage
    col_nombre_geo = buscar_columna(gdf_deptos, ['departamento', 'nam', 'nombre', 'partido', 'dpto'])
    col_prov_geo   = buscar_columna(gdf_deptos, ['provincia'])

    if col_nombre_geo is None:
        st.error(f"❌ No se detectó columna de nombre en el GeoPackage. Columnas: {list(gdf_deptos.columns)}")
    else:
        # Normalizar nombres para el match
        gdf_deptos['_dept_norm'] = gdf_deptos[col_nombre_geo].apply(normalizar_nombre)

        # Unir con clusters
        df_clusters = df_med[['PROVINCIA', 'DEPARTAMENTO', '_dept_norm', 'cluster']].drop_duplicates('_dept_norm')
        gdf_deptos = gdf_deptos.merge(
            df_clusters[['_dept_norm', 'cluster']],
            on='_dept_norm',
            how='left'
        )
        gdf_deptos['cluster'] = gdf_deptos['cluster'].fillna(-1).astype(int)

        # Calcular centro del mapa
        try:
            centro = gdf_deptos.geometry.unary_union.centroid
            centro_lat, centro_lon = centro.y, centro.x
        except Exception:
            centro_lat, centro_lon = -36.0, -60.0

        # Crear mapa
        m = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=5,
            tiles="OpenStreetMap"
        )

        # Función de estilo
        def estilo_depto(feature):
            cluster = feature['properties'].get('cluster', -1)
            color = COLORES_CLUSTER.get(cluster, '#cccccc')
            return {
                'fillColor': color,
                'color': 'black',
                'weight': 0.8,
                'fillOpacity': 0.65,
            }

        # Agregar capa GeoJSON
        folium.GeoJson(
            gdf_deptos.__geo_interface__,
            name='Departamentos',
            style_function=estilo_depto,
            tooltip=folium.GeoJsonTooltip(
                fields=[col_nombre_geo],
                aliases=['Departamento:'],
                localize=True,
            ),
        ).add_to(m)

        # Renderizar y capturar clic
        map_data = st_folium(
            m,
            width="100%",
            height=500,
            returned_objects=["last_active_drawing"],
            key="mapa_deptos",
        )

        # Procesar clic
        if map_data and map_data.get("last_active_drawing"):
            props = map_data["last_active_drawing"]["properties"]
            clicked_name = props.get(col_nombre_geo, "")
            clicked_norm = normalizar_nombre(clicked_name)

            # Buscar el nombre real en df_med
            match_row = df_med[df_med['_dept_norm'] == clicked_norm]
            if not match_row.empty:
                nuevo_depto = match_row['DEPARTAMENTO'].iloc[0]
                nueva_prov  = match_row['PROVINCIA'].iloc[0]
                if st.session_state["depto_sel"] != nuevo_depto:
                    st.session_state["depto_sel"] = nuevo_depto
                    st.session_state["prov_sel"] = nueva_prov
                    st.rerun()
else:
    st.info("ℹ️ Subí el archivo 'Departamentos_area_estudio.gpkg' al repositorio para ver el mapa.")

# ─────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtrado de Datos")

provincias = sorted(df_med['PROVINCIA'].dropna().unique())

# Sincronizar prov_sel con session_state
if st.session_state["prov_sel"] not in provincias:
    st.session_state["prov_sel"] = provincias[0] if provincias else None

prov_sel = st.sidebar.selectbox(
    "Provincia",
    provincias,
    key="prov_sel"
)
prov_norm = normalizar_nombre(prov_sel)

df_med_prov = df_med[df_med['_prov_norm'] == prov_norm]
deptos = sorted(df_med_prov['DEPARTAMENTO'].dropna().unique())

# Sincronizar depto_sel con session_state
if st.session_state["depto_sel"] not in deptos:
    st.session_state["depto_sel"] = deptos[0] if deptos else None

depto_sel = st.sidebar.selectbox(
    "Departamento",
    deptos,
    key="depto_sel"
)
depto_norm = normalizar_nombre(depto_sel)

# Cruce robusto con emergencia
mask_eme = (df_eme['_prov_norm'] == prov_norm) & (df_eme['_dept_norm'] == depto_norm)
df_eme_depto = df_eme[mask_eme]
if df_eme_depto.empty:
    df_eme_depto = df_eme[df_eme['_dept_norm'] == depto_norm]
if df_eme_depto.empty:
    deptos_eme_unicos = df_eme[['DEPARTAMENTO', '_dept_norm']].drop_duplicates()
    deptos_eme_dict = dict(zip(deptos_eme_unicos['DEPARTAMENTO'], deptos_eme_unicos['_dept_norm']))
    match = match_departamento(depto_sel, deptos_eme_dict)
    if match:
        df_eme_depto = df_eme[df_eme['DEPARTAMENTO'] == match]

if df_eme_depto.empty:
    actividades = []
    st.sidebar.warning("⚠️ No se encontraron emergencias para este departamento.")
else:
    actividades = sorted(df_eme_depto['ACTIVIDAD'].dropna().unique())

act_sel = st.sidebar.multiselect(
    "Selección de Actividades",
    actividades,
    default=actividades if actividades else []
)

# ─────────────────────────────────────────────────────────────────────
# DATOS DE SEQUÍA (MEDIANA Y MÁXIMO)
# ─────────────────────────────────────────────────────────────────────
fila_med = df_med_prov[df_med_prov['DEPARTAMENTO'] == depto_sel]
if fila_med.empty:
    st.warning(f"⚠️ No hay datos de sequía (mediana) para {depto_sel} ({prov_sel}).")
    st.stop()

valores_med = []
for p in periodos_todos:
    if p in periodos_med:
        v = fila_med[p].iloc[0]
        valores_med.append(v if pd.notna(v) else np.nan)
    else:
        valores_med.append(np.nan)

df_max_prov = df_max[df_max['_prov_norm'] == prov_norm]
fila_max = df_max_prov[df_max_prov['DEPARTAMENTO'] == depto_sel]
valores_max = []
if fila_max.empty:
    valores_max = [np.nan] * len(periodos_todos)
else:
    for p in periodos_todos:
        if p in periodos_med:
            v = fila_max[p].iloc[0]
            valores_max.append(v if pd.notna(v) else np.nan)
        else:
            valores_max.append(np.nan)

# ─────────────────────────────────────────────────────────────────────
# EMERGENCIAS
# ─────────────────────────────────────────────────────────────────────
emergencias_por_act = {}
for act in act_sel:
    emergencias_por_act[act] = {}

if act_sel and not df_eme_depto.empty:
    df_eme_filt = df_eme_depto[df_eme_depto['ACTIVIDAD'].isin(act_sel)]
    for _, row in df_eme_filt.iterrows():
        act = row['ACTIVIDAD']
        for p in periodos_todos:
            if p in periodos_eme:
                val = str(row[p]).strip()
                if val and val.lower() not in ('nan', 'none', '', 'null'):
                    if p not in emergencias_por_act[act]:
                        emergencias_por_act[act][p] = val

# ─────────────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────────────
st.subheader(f"Evolución de la sequía — {depto_sel} ({prov_sel})")

n_acts = len(act_sel)

if n_acts == 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periodos_todos,
        y=valores_med,
        mode='lines+markers',
        name='Mediana departamental',
        line=dict(color='black', width=2.5),
        marker=dict(size=8, color='black'),
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=periodos_todos,
        y=valores_max,
        mode='lines+markers',
        name='Máximo departamental',
        line=dict(color='#808080', width=2, dash='dot'),
        marker=dict(size=7, color='#808080', symbol='diamond'),
        connectgaps=False,
    ))
    fig.add_hline(
        y=6,
        line_dash="dash",
        line_color="red",
        line_width=1.8,
        annotation_text="Límite sequía (6)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(
            title="Mes",
            tickangle=-45,
            tickmode='array',
            tickvals=list(range(len(periodos_todos))),
            ticktext=periodos_todos,
            range=[-0.5, len(periodos_todos) - 0.5],
        ),
        yaxis=dict(title="Valor acumulado de sequía (0-9)", range=[-0.5, 9.5], dtick=1),
        template='plotly_white',
        height=500,
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
    )
else:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("", "Resoluciones Nacionales de Emergencia por tipo de Actividad"),
    )

    fig.add_trace(go.Scatter(
        x=periodos_todos,
        y=valores_med,
        mode='lines+markers',
        name='Mediana departamental',
        line=dict(color='black', width=2.5),
        marker=dict(size=8, color='black'),
        connectgaps=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=periodos_todos,
        y=valores_max,
        mode='lines+markers',
        name='Máximo departamental',
        line=dict(color='#808080', width=2, dash='dot'),
        marker=dict(size=7, color='#808080', symbol='diamond'),
        connectgaps=False,
    ), row=1, col=1)

    fig.add_hline(
        y=6,
        line_dash="dash",
        line_color="red",
        line_width=1.8,
        annotation_text="Límite del protocolo de sequía (6)",
        annotation_position="top right",
        row=1, col=1,
    )

    for idx, act in enumerate(act_sel):
        color = color_para_actividad(act)
        emergencias_act = emergencias_por_act[act]
        if not emergencias_act:
            continue
        xs = list(emergencias_act.keys())
        ys = [idx] * len(xs)
        hover_text = [
            f"<b>{act}</b><br>Mes: {x}<br>Resolución: {emergencias_act[x]}"
            for x in xs
        ]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            name=act,
            marker=dict(symbol='square', size=22, color=color,
                        line=dict(color='white', width=1)),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True,
        ), row=2, col=1)

    fig.update_xaxes(
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        row=1, col=1,
    )
    fig.update_xaxes(
        title="Mes",
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        row=2, col=1,
    )
    fig.update_yaxes(
        title="Valor acumulado de sequía (0-9)",
        range=[-0.5, 9.5],
        dtick=1,
        row=1, col=1,
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(n_acts)),
        ticktext=act_sel,
        range=[-0.5, n_acts - 0.5],
        row=2, col=1,
    )

    fig.update_layout(
        template='plotly_white',
        height=680,
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'),
    )

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TABLA DE RESOLUCIONES
# ─────────────────────────────────────────────────────────────────────
st.subheader("Resoluciones de emergencia declaradas")

filas = []
for act in act_sel:
    for p, res in emergencias_por_act[act].items():
        filas.append({
            'Mes': p,
            'Actividad': act,
            'Resolución': res,
            'Intensidad (mediana)': valores_med[periodos_todos.index(p)],
            'Intensidad (máximo)': valores_max[periodos_todos.index(p)],
        })

if filas:
    df_tabla = pd.DataFrame(filas)
    df_tabla['_orden'] = df_tabla['Mes'].apply(lambda m: periodos_todos.index(m))
    df_tabla = df_tabla.sort_values('_orden').drop(columns='_orden')
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)
else:
    if not actividades:
        st.warning(
            f"⚠️ No se encontraron actividades para **{depto_sel}** en la base de emergencia. "
            "Revisá el expander **'🔧 Ver datos crudos'** para diagnosticar."
        )
    else:
        st.info("ℹ️ No hay resoluciones de emergencia para esta selección.")

# ─────────────────────────────────────────────────────────────────────
# EXPANDER DE DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Ver datos crudos (diagnóstico del cruce)"):
    st.write("### Base de sequía (mediana)")
    st.write("**Provincias disponibles:**", sorted(df_med['PROVINCIA'].unique())[:20])
    st.write(f"**Departamentos de '{prov_sel}':**",
             sorted(df_med_prov['DEPARTAMENTO'].unique())[:30])

    st.write("### Base de sequía (máximo)")
    st.write(f"**Departamentos de '{prov_sel}':**",
             sorted(df_max_prov['DEPARTAMENTO'].unique())[:30])

    st.write("### Base de emergencia")
    st.write("**Provincias disponibles:**", sorted(df_eme['PROVINCIA'].unique())[:20])
    st.write(f"**Departamentos de '{prov_sel}':**",
             sorted(df_eme[df_eme['_prov_norm'] == prov_norm]['DEPARTAMENTO'].unique())[:30])

    st.write("### Cruce actual")
    st.write(f"Departamento seleccionado: `{depto_sel}`")
    st.write(f"Filas en emergencia: **{len(df_eme_depto)}**")
    if len(df_eme_depto) > 0:
        st.write("Actividades detectadas:", sorted(df_eme_depto['ACTIVIDAD'].unique()))

    st.write("### Fechas")
    st.write(f"**Períodos sequía ({len(periodos_med)}):**", periodos_med)
    st.write(f"**Períodos emergencia ({len(periodos_eme)}):**", periodos_eme)
    st.write(f"**Períodos totales ({len(periodos_todos)}):**", periodos_todos)

    if GEOPACKAGE_OK and gdf_deptos is not None:
        st.write("### GeoPackage")
        st.write("**Columnas:**", list(gdf_deptos.columns))
        st.write("**Cantidad de features:**", len(gdf_deptos))
        st.write("**CRS:**", gdf_deptos.crs)
