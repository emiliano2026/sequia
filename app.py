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
# PALETA DE SEQUÍA (0-9)
# ─────────────────────────────────────────────────────────────────────
PALETA_SEQUIA = {
    0: '#FFFFFF',
    1: '#FFFFCC',
    2: '#FFEDA0',
    3: '#FED976',
    4: '#FEB24C',
    5: '#FD8D3C',
    6: '#FC4E2A',
    7: '#E31A1C',
    8: '#BD0026',
    9: '#800026',
}

def color_sequia(valor):
    if pd.isna(valor):
        return '#E0E0E0'
    try:
        return PALETA_SEQUIA.get(int(round(float(valor))), '#999999')
    except (ValueError, TypeError):
        return '#999999'

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

    # Base de sequía MEDIANA
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

    # Base de sequía MÁXIMO
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

    # Base de emergencia
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

    periodos_todos = list(periodos_med)
    for p in periodos_eme:
        if p not in periodos_todos:
            periodos_todos.append(p)

    return df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos

df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos = load_data()

# ─────────────────────────────────────────────────────────────────────
# CARGA DEL GEOPACKAGE (cacheada y simplificada para rendimiento)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_geopackage():
    import geopandas as gpd
    try:
        gdf = gpd.read_file("Departamentos_area_estudio.gpkg")
    except Exception:
        url = "https://raw.githubusercontent.com/emiliano2026/sequia/main/Departamentos_area_estudio.gpkg"
        gdf = gpd.read_file(url)

    if gdf.crs is not None and gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Simplificar geometría para acelerar el render (0.005° ≈ 500m)
    gdf['geometry'] = gdf.geometry.simplify(0.005, preserve_topology=True)

    # Detectar columna de nombre
    col_nombre = None
    for patron in ['departamento', 'nam', 'nombre', 'partido', 'dpto']:
        for c in gdf.columns:
            if normalizar_nombre(patron) in normalizar_nombre(c):
                col_nombre = c
                break
        if col_nombre:
            break

    gdf['_dept_norm'] = gdf[col_nombre].apply(normalizar_nombre)

    # Calcular centroide una sola vez
    try:
        centro = gdf.geometry.unary_union.centroid
        centro_lat, centro_lon = centro.y, centro.x
    except Exception:
        centro_lat, centro_lon = -36.0, -60.0

    return gdf, col_nombre, (centro_lat, centro_lon)

try:
    gdf_deptos, col_nombre_geo, centro_mapa = cargar_geopackage()
    GEOPACKAGE_OK = True
except Exception as e:
    st.sidebar.warning(f"⚠️ No se pudo cargar el GeoPackage: {e}")
    gdf_deptos = None
    col_nombre_geo = None
    centro_mapa = (-36.0, -60.0)
    GEOPACKAGE_OK = False

# ─────────────────────────────────────────────────────────────────────
# MAPA COROPLÉTICO INTERACTIVO
# ─────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Distribución espacial de la sequía")

if GEOPACKAGE_OK and gdf_deptos is not None and col_nombre_geo is not None:

    # ── Selectores (a la izquierda) y mapa (a la derecha) ──────────
    col_sel, col_mapa = st.columns([1, 4])

    with col_sel:
        st.markdown("**Período**")
        periodo_mapa = st.selectbox(
            "Mes",
            periodos_med,
            index=0,
            key="periodo_mapa",
            label_visibility="collapsed",
        )

        st.markdown("**Estadístico**")
        estadistico_mapa = st.radio(
            "Estadístico",
            ["Mediana", "Máximo"],
            key="estadistico_mapa",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Escala**")
        # Leyenda vertical
        leyenda_items = ""
        for i in range(10):
            leyenda_items += (
                f'<div style="display:flex; align-items:center; margin-bottom:2px;">'
                f'<div style="width:28px; height:20px; background:{PALETA_SEQUIA[i]}; '
                f'border:1px solid #999;"></div>'
                f'<span style="margin-left:8px; font-size:13px;">{i}</span>'
                f'</div>'
            )
        leyenda_items += (
            f'<div style="display:flex; align-items:center; margin-top:4px;">'
            f'<div style="width:28px; height:20px; background:#E0E0E0; '
            f'border:1px solid #999;"></div>'
            f'<span style="margin-left:8px; font-size:13px;">Sin dato</span>'
            f'</div>'
        )
        st.markdown(leyenda_items, unsafe_allow_html=True)

    with col_mapa:
        # Elegir base según estadístico
        df_sel = df_med if estadistico_mapa == "Mediana" else df_max

        # Merge: obtener el valor del mes seleccionado por departamento
        df_val = df_sel[['_dept_norm', periodo_mapa]].copy()
        df_val = df_val.rename(columns={periodo_mapa: 'valor'})
        df_val = df_val.dropna(subset=['valor']).drop_duplicates('_dept_norm')

        gdf_plot = gdf_deptos.merge(df_val, on='_dept_norm', how='left')

        # Crear mapa
        m = folium.Map(
            location=list(centro_mapa),
            zoom_start=5,
            tiles="CartoDB positron",
        )

        def estilo(feature):
            valor = feature['properties'].get('valor')
            return {
                'fillColor': color_sequia(valor),
                'color': 'black',
                'weight': 0.6,
                'fillOpacity': 0.85,
            }

        # Preparar los datos de las features: necesitamos 'valor' en properties
        geojson_data = gdf_plot.__geo_interface__
        for feat, (_, row) in zip(geojson_data['features'], gdf_plot.iterrows()):
            feat['properties']['valor'] = None if pd.isna(row['valor']) else float(row['valor'])

        folium.GeoJson(
            geojson_data,
            name='Sequía',
            style_function=estilo,
            tooltip=folium.GeoJsonTooltip(
                fields=[col_nombre_geo, 'valor'],
                aliases=['Departamento:', f'{estadistico_mapa} {periodo_mapa}:'],
                localize=True,
            ),
        ).add_to(m)

        st_folium(m, width=None, height=520, key="mapa_sequia")

    st.caption(
        f"Valor {estadistico_mapa.lower()} de intensidad de sequía acumulada — {periodo_mapa}. "
        "Pasá el mouse sobre cada departamento para ver su valor."
    )

else:
    st.info("ℹ️ Subí el archivo 'Departamentos_area_estudio.gpkg' al repositorio para ver el mapa.")

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "depto_sel" not in st.session_state:
    st.session_state["depto_sel"] = None
if "prov_sel" not in st.session_state:
    st.session_state["prov_sel"] = None

# ─────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtrado de Datos")

provincias = sorted(df_med['PROVINCIA'].dropna().unique())
if st.session_state["prov_sel"] not in provincias:
    st.session_state["prov_sel"] = provincias[0] if provincias else None

prov_sel = st.sidebar.selectbox("Provincia", provincias, key="prov_sel")
prov_norm = normalizar_nombre(prov_sel)

df_med_prov = df_med[df_med['_prov_norm'] == prov_norm]
deptos = sorted(df_med_prov['DEPARTAMENTO'].dropna().unique())
if st.session_state["depto_sel"] not in deptos:
    st.session_state["depto_sel"] = deptos[0] if deptos else None

depto_sel = st.sidebar.selectbox("Departamento", deptos, key="depto_sel")
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
        x=periodos_todos, y=valores_med, mode='lines+markers',
        name='Mediana departamental',
        line=dict(color='black', width=2.5),
        marker=dict(size=8, color='black'),
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=periodos_todos, y=valores_max, mode='lines+markers',
        name='Máximo departamental',
        line=dict(color='#808080', width=2, dash='dot'),
        marker=dict(size=7, color='#808080', symbol='diamond'),
        connectgaps=False,
    ))
    fig.add_hline(
        y=6, line_dash="dash", line_color="red", line_width=1.8,
        annotation_text="Límite sequía (6)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(
            title="Mes", tickangle=-45, tickmode='array',
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
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=("", "Resoluciones Nacionales de Emergencia por tipo de Actividad"),
    )

    fig.add_trace(go.Scatter(
        x=periodos_todos, y=valores_med, mode='lines+markers',
        name='Mediana departamental',
        line=dict(color='black', width=2.5),
        marker=dict(size=8, color='black'),
        connectgaps=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=periodos_todos, y=valores_max, mode='lines+markers',
        name='Máximo departamental',
        line=dict(color='#808080', width=2, dash='dot'),
        marker=dict(size=7, color='#808080', symbol='diamond'),
        connectgaps=False,
    ), row=1, col=1)

    fig.add_hline(
        y=6, line_dash="dash", line_color="red", line_width=1.8,
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
            x=xs, y=ys, mode='markers', name=act,
            marker=dict(symbol='square', size=22, color=color,
                        line=dict(color='white', width=1)),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True,
        ), row=2, col=1)

    fig.update_xaxes(
        tickangle=-45, tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        row=1, col=1,
    )
    fig.update_xaxes(
        title="Mes", tickangle=-45, tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        row=2, col=1,
    )
    fig.update_yaxes(
        title="Valor acumulado de sequía (0-9)",
        range=[-0.5, 9.5], dtick=1, row=1, col=1,
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(n_acts)),
        ticktext=act_sel,
        range=[-0.5, n_acts - 0.5],
        row=2, col=1,
    )

    fig.update_layout(
        template='plotly_white', height=680,
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
        st.write("**Columna de nombre detectada:**", col_nombre_geo)
        st.write("**Cantidad de features:**", len(gdf_deptos))
        st.write("**CRS:**", gdf_deptos.crs)
