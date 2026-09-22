import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import unicodedata
import json
import urllib.request
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
        El valor de sequía corresponde al valor acumulado trimestral, sumatoria de los valores de intensidad de sequía: leve = 1, moderada = 2 y severa = 3, información generada por la Mesa Nacional de Monitoreo de Sequías (MNMS). Se representan valores agrupados por departamentos: <b>mediana</b> y <b>máximo</b>.
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

def construir_key(provincia, departamento):
    return f"{normalizar_nombre(provincia)}|{normalizar_nombre(departamento)}"

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
    0: '#FFFFFF', 1: '#FFFFCC', 2: '#FFEDA0', 3: '#FED976', 4: '#FEB24C',
    5: '#FD8D3C', 6: '#FC4E2A', 7: '#E31A1C', 8: '#BD0026', 9: '#800026',
}

def color_sequia(valor):
    if pd.isna(valor):
        return '#E0E0E0'
    try:
        return PALETA_SEQUIA.get(int(round(float(valor))), '#999999')
    except (ValueError, TypeError):
        return '#999999'

# ─────────────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSIÓN — CÁLCULO
# ─────────────────────────────────────────────────────────────────────
def matriz_confusion_depto(key, df_seq, df_eme, periodos, periodos_eme,
                           umbral=6, delay=1):
    """Calcula la matriz de confusión mes a mes para un departamento."""
    fila = df_seq[df_seq['_key'] == key]
    if fila.empty:
        return None

    valores = [fila[p].iloc[0] if p in periodos else np.nan for p in periodos]
    df_eme_key = df_eme[df_eme['_key'] == key]

    tiene_evento = [pd.notna(v) and v >= umbral for v in valores]

    tiene_respuesta_mes = []
    for p in periodos:
        hay = False
        for _, row in df_eme_key.iterrows():
            val = str(row[p]).strip() if p in periodos_eme else ''
            if val and val.lower() not in ('nan', 'none', '', 'null'):
                hay = True
                break
        tiene_respuesta_mes.append(hay)

    tiene_respuesta = tiene_respuesta_mes.copy()
    if delay > 0:
        tiene_respuesta = [False] * len(periodos)
        for i in range(len(periodos)):
            if tiene_respuesta_mes[i]:
                for j in range(max(0, i - delay), i + 1):
                    tiene_respuesta[j] = True

    VP = sum(1 for e, r in zip(tiene_evento, tiene_respuesta) if e and r)
    FN = sum(1 for e, r in zip(tiene_evento, tiene_respuesta) if e and not r)
    FP = sum(1 for e, r in zip(tiene_evento, tiene_respuesta) if not e and r)
    VN = sum(1 for e, r in zip(tiene_evento, tiene_respuesta) if not e and not r)

    return {
        'key': key,
        'provincia': fila['PROVINCIA'].iloc[0],
        'departamento': fila['DEPARTAMENTO'].iloc[0],
        'VP': VP, 'FN': FN, 'FP': FP, 'VN': VN,
    }


def calcular_metricas(VP, FN, FP, VN):
    """Calcula las métricas de desempeño a partir de la matriz de confusión."""
    total = VP + FN + FP + VN
    return {
        'EXACTITUD':     (VP + VN) / total if total > 0 else np.nan,
        'PRECISIÓN':     VP / (VP + FP) if (VP + FP) > 0 else np.nan,
        'SENSIBILIDAD':  VP / (VP + FN) if (VP + FN) > 0 else np.nan,
        'ESPECIFICIDAD': VN / (VN + FP) if (VN + FP) > 0 else np.nan,
    }


def calcular_tiempo_respuesta(key, df_seq, df_eme, periodos, periodos_eme,
                              umbral=6, ventana_max=6):
    """
    Calcula el tiempo de respuesta (en meses) entre el inicio de cada evento
    de sequía y la primera resolución declarada (cualquier actividad).

    Returns:
        dict con:
            - tiempos: lista de delays por evento respondido
            - promedio: promedio de los delays (o None)
            - minimo: mínimo (o None)
            - maximo: máximo (o None)
            - eventos_totales: cantidad total de eventos detectados
            - eventos_respondidos: cantidad de eventos con respuesta dentro de la ventana
    """
    fila = df_seq[df_seq['_key'] == key]
    if fila.empty:
        return None

    valores = [fila[p].iloc[0] if p in periodos else np.nan for p in periodos]
    df_eme_key = df_eme[df_eme['_key'] == key]

    # Detectar eventos como rachas de valores >= umbral
    eventos = []
    en_evento = False
    inicio = None
    for i, v in enumerate(valores):
        cumple = pd.notna(v) and v >= umbral
        if cumple and not en_evento:
            en_evento = True
            inicio = i
        elif not cumple and en_evento:
            en_evento = False
            eventos.append({'inicio': inicio, 'fin': i - 1})
    if en_evento:
        eventos.append({'inicio': inicio, 'fin': len(valores) - 1})

    # ¿En qué meses hay alguna resolución (cualquier actividad)?
    meses_con_respuesta = set()
    for _, row in df_eme_key.iterrows():
        for i, p in enumerate(periodos):
            if p in periodos_eme:
                val = str(row[p]).strip()
                if val and val.lower() not in ('nan', 'none', '', 'null'):
                    meses_con_respuesta.add(i)

    # Para cada evento, buscar la primera resolución dentro de la ventana
    tiempos = []
    for ev in eventos:
        i0 = ev['inicio']
        i1 = min(ev['fin'] + ventana_max, len(periodos) - 1)
        # Buscar el primer mes con respuesta en [i0, i1]
        for i in range(i0, i1 + 1):
            if i in meses_con_respuesta:
                tiempos.append(i - i0)
                break

    return {
        'tiempos': tiempos,
        'promedio': float(np.mean(tiempos)) if tiempos else None,
        'minimo': int(np.min(tiempos)) if tiempos else None,
        'maximo': int(np.max(tiempos)) if tiempos else None,
        'eventos_totales': len(eventos),
        'eventos_respondidos': len(tiempos),
    }


# ─────────────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSIÓN — VISUALIZACIÓN (modo combinado, fuentes grandes)
# ─────────────────────────────────────────────────────────────────────
def matriz_confusion_plotly(VP, FN, FP, VN, titulo="Matriz de confusión", height=460):
    """
    Dibuja la matriz de confusión con:
      - Modo combinado: valor absoluto + porcentaje sobre el total
      - Paleta semáforo pastel (verde=bueno, rojo=malo, naranja=alerta)
      - Fuentes grandes
    """
    fig = go.Figure()

    total = VP + FN + FP + VN
    if total == 0:
        total = 1
    max_val = max(VP, FN, FP, VN) or 1

    def color_semaforo(base_rgb, valor):
        """Mezcla el color base con blanco (60% - 100% de saturación)."""
        intensidad = 0.6 + 0.4 * (valor / max_val)
        r = int(255 * (1 - intensidad) + base_rgb[0] * intensidad)
        g = int(255 * (1 - intensidad) + base_rgb[1] * intensidad)
        b = int(255 * (1 - intensidad) + base_rgb[2] * intensidad)
        return f"rgb({r},{g},{b})"

    color_VP = color_semaforo((20, 150, 60), VP)     # verde
    color_FN = color_semaforo((200, 30, 30), FN)     # rojo
    color_FP = color_semaforo((240, 140, 30), FP)    # naranja
    color_VN = color_semaforo((140, 200, 140), VN)   # verde claro

    # Definir celdas: (x0, x1, y0, y1, valor, etiqueta, color)
    celdas = [
        (-0.5, 0.5, -0.5, 0.5, VP, "Verdaderos Positivos<br>(VP)", color_VP),
        ( 0.5, 1.5, -0.5, 0.5, FN, "Falsos Negativos<br>(FN)", color_FN),
        (-0.5, 0.5,  0.5, 1.5, FP, "Falsos Positivos<br>(FP)", color_FP),
        ( 0.5, 1.5,  0.5, 1.5, VN, "Verdaderos Negativos<br>(VN)", color_VN),
    ]

    for x0, x1, y0, y1, valor, etiqueta, color in celdas:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color,
            line=dict(color='white', width=4),
            layer='below',
        )
        pct = valor / total * 100
        # Texto: etiqueta arriba, valor grande, porcentaje abajo
        texto = (
            f"<span style='font-size:16px'>{etiqueta}</span>"
            f"<br><b style='font-size:34px'>{valor}</b>"
            f"<br><span style='font-size:16px'>({pct:.1f}%)</span>"
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=texto,
            showarrow=False,
            font=dict(size=16, color='black'),
            align='center',
        )

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=['Hay respuesta', 'No hay respuesta'],
        range=[-0.5, 1.5],
        side='top',
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=15),
    )
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=['Hay evento', 'No hay evento'],
        range=[1.5, -0.5],
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=15),
    )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center',
                   font=dict(size=18)),
        height=height,
        margin=dict(l=140, r=30, t=90, b=30),
        plot_bgcolor='white',
        showlegend=False,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (CSV)
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

    # MEDIANA
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
    df_med['_key']         = df_med.apply(lambda r: construir_key(r['PROVINCIA'], r['DEPARTAMENTO']), axis=1)

    # MÁXIMO
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
    df_max['_key']         = df_max.apply(lambda r: construir_key(r['PROVINCIA'], r['DEPARTAMENTO']), axis=1)

    # EMERGENCIA
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
    df_eme['_key']         = df_eme.apply(lambda r: construir_key(r['PROVINCIA'], r['DEPARTAMENTO']), axis=1)

    periodos_todos = list(periodos_med)
    for p in periodos_eme:
        if p not in periodos_todos:
            periodos_todos.append(p)

    return df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos

df_med, df_max, df_eme, periodos_med, periodos_eme, periodos_todos = load_data()

# ─────────────────────────────────────────────────────────────────────
# CARGA DEL GEOJSON
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_geojson():
    url = "https://raw.githubusercontent.com/emiliano2026/sequia/main/Departamentos_area_estudio_v3.geojson"
    with urllib.request.urlopen(url) as resp:
        geojson_data = json.loads(resp.read().decode('utf-8'))

    col_nombre = 'DEPARTAMENTO'
    props_ejemplo = geojson_data['features'][0]['properties']
    col_provincia_geo = None
    for key in props_ejemplo.keys():
        if 'provincia' in normalizar_nombre(key):
            col_provincia_geo = key
            break

    for feat in geojson_data['features']:
        props = feat['properties']
        nombre = props.get(col_nombre, '')
        if col_provincia_geo:
            provincia = props.get(col_provincia_geo, '')
            key = construir_key(provincia, nombre)
        else:
            key = f"buenosaires|{normalizar_nombre(nombre)}"
        props['_key'] = key
        props['_dept_norm'] = normalizar_nombre(nombre)

    lats, lons = [], []
    for feat in geojson_data['features']:
        coords = feat['geometry']['coordinates']
        def recorrer(c):
            if isinstance(c[0], (int, float)):
                lons.append(c[0]); lats.append(c[1])
            else:
                for x in c: recorrer(x)
        recorrer(coords)
    centro = (np.mean(lats), np.mean(lons)) if lats else (-36.0, -60.0)

    return geojson_data, col_nombre, col_provincia_geo, centro

try:
    geojson_deptos, col_nombre_geo, col_provincia_geo, centro_mapa = cargar_geojson()
    GEOJSON_OK = True
except Exception as e:
    st.sidebar.warning(f"⚠️ No se pudo cargar el GeoJSON: {e}")
    geojson_deptos = None
    col_nombre_geo = None
    col_provincia_geo = None
    centro_mapa = (-36.0, -60.0)
    GEOJSON_OK = False

# ─────────────────────────────────────────────────────────────────────
# CARGA DE MATRICES PRECALCULADAS (CSV en GitHub)
# ─────────────────────────────────────────────────────────────────────
URL_MC_DEPTO_MED = "https://raw.githubusercontent.com/emiliano2026/sequia/main/matriz_confusion_depto.csv"
URL_MC_PROV_MED  = "https://raw.githubusercontent.com/emiliano2026/sequia/main/matriz_confusion_provincia.csv"
URL_MC_DEPTO_MAX = "https://raw.githubusercontent.com/emiliano2026/sequia/main/matriz_confusion_depto_max.csv"
URL_MC_PROV_MAX  = "https://raw.githubusercontent.com/emiliano2026/sequia/main/matriz_confusion_provincia_max.csv"

@st.cache_data
def cargar_matrices_precalculadas():
    """Carga las matrices precalculadas. Devuelve None si algún CSV no existe."""
    def safe_read(url, dtype=None):
        try:
            return pd.read_csv(url, dtype=dtype)
        except Exception:
            return None
    return (
        safe_read(URL_MC_DEPTO_MED, dtype={'key': str}),
        safe_read(URL_MC_PROV_MED),
        safe_read(URL_MC_DEPTO_MAX, dtype={'key': str}),
        safe_read(URL_MC_PROV_MAX),
    )

df_mc_depto_med, df_mc_prov_med, df_mc_depto_max, df_mc_prov_max = cargar_matrices_precalculadas()

def obtener_matriz_depto(key, estadistico="Mediana"):
    """Devuelve la matriz del departamento desde el CSV correspondiente o la calcula."""
    if estadistico == "Mediana" and df_mc_depto_med is not None:
        fila = df_mc_depto_med[df_mc_depto_med['key'] == key]
    elif estadistico == "Máximo" and df_mc_depto_max is not None:
        fila = df_mc_depto_max[df_mc_depto_max['key'] == key]
    else:
        fila = pd.DataFrame()

    if not fila.empty:
        return {
            'key': key,
            'provincia': fila['PROVINCIA'].iloc[0],
            'departamento': fila['DEPARTAMENTO'].iloc[0],
            'VP': int(fila['VP'].iloc[0]),
            'FN': int(fila['FN'].iloc[0]),
            'FP': int(fila['FP'].iloc[0]),
            'VN': int(fila['VN'].iloc[0]),
        }
    # Fallback: calcular al vuelo
    df_src = df_med if estadistico == "Mediana" else df_max
    return matriz_confusion_depto(key, df_src, df_eme, periodos_med, periodos_eme,
                                   umbral=6, delay=1)

# ─────────────────────────────────────────────────────────────────────
# SINCRONIZACIÓN DE FILTROS
# ─────────────────────────────────────────────────────────────────────
provincias = sorted(df_med['PROVINCIA'].dropna().unique())

if "prov_sel" not in st.session_state or st.session_state["prov_sel"] not in provincias:
    st.session_state["prov_sel"] = provincias[0] if provincias else None

df_med_prov_actual = df_med[df_med['_prov_norm'] == normalizar_nombre(st.session_state["prov_sel"])]
deptos_actuales = sorted(df_med_prov_actual['DEPARTAMENTO'].dropna().unique())

if "depto_sel" not in st.session_state or st.session_state["depto_sel"] not in deptos_actuales:
    st.session_state["depto_sel"] = deptos_actuales[0] if deptos_actuales else None

if st.session_state["prov_sel"] and st.session_state["depto_sel"]:
    st.session_state["key_sel"] = construir_key(
        st.session_state["prov_sel"], st.session_state["depto_sel"]
    )
else:
    st.session_state["key_sel"] = None

key_actual = st.session_state["key_sel"]

# ─────────────────────────────────────────────────────────────────────
# MAPA COROPLÉTICO
# ─────────────────────────────────────────────────────────────────────
st.subheader("Distribución espacial de la sequía en la región Noreste y Centro")
st.caption("Clic en un departamento del mapa para graficar sus curvas")

if GEOJSON_OK and geojson_deptos is not None:

    col_sel, col_mapa = st.columns([1, 4])

    with col_sel:
        st.markdown("**Seleccionar mes**")
        periodo_mapa = st.selectbox(
            "Mes", periodos_med, index=0,
            key="periodo_mapa", label_visibility="collapsed",
        )

        st.markdown("**Seleccionar estadístico**")
        estadistico_mapa = st.radio(
            "Estadístico", ["Mediana", "Máximo"],
            key="estadistico_mapa", label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Escala**")
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
        df_sel = df_med if estadistico_mapa == "Mediana" else df_max
        df_val = df_sel[['_key', periodo_mapa]].copy()
        df_val = df_val.rename(columns={periodo_mapa: 'valor'})
        df_val = df_val.dropna(subset=['valor']).drop_duplicates('_key')
        valores_por_key = dict(zip(df_val['_key'], df_val['valor']))

        m = folium.Map(
            location=list(centro_mapa),
            zoom_start=5,
            tiles=None,
        )
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
            overlay=False,
            control=True,
        ).add_to(m)

        def estilo(feature):
            key = feature['properties'].get('_key', '')
            valor = valores_por_key.get(key)
            es_sel = (key == key_actual) and key != ""
            return {
                'fillColor': color_sequia(valor),
                'color': '#0000FF' if es_sel else 'black',
                'weight': 3 if es_sel else 0.6,
                'fillOpacity': 0.9 if es_sel else 0.85,
            }

        if col_provincia_geo:
            campos_tooltip = [col_provincia_geo, col_nombre_geo]
            alias_tooltip = ['Provincia:', 'Departamento:']
        else:
            campos_tooltip = [col_nombre_geo]
            alias_tooltip = ['Departamento:']

        folium.GeoJson(
            geojson_deptos,
            name='Sequía',
            style_function=estilo,
            tooltip=folium.GeoJsonTooltip(
                fields=campos_tooltip,
                aliases=alias_tooltip,
                localize=True,
            ),
        ).add_to(m)

        map_data = st_folium(
            m,
            width=None,
            height=520,
            key="mapa_sequia",
            returned_objects=["last_active_drawing"],
        )

    if map_data and map_data.get("last_active_drawing"):
        props = map_data["last_active_drawing"]["properties"]
        clicked_key = props.get("_key", "")

        if clicked_key and clicked_key != key_actual:
            match_row = df_med[df_med['_key'] == clicked_key]
            if not match_row.empty:
                st.session_state["prov_sel"]   = match_row['PROVINCIA'].iloc[0]
                st.session_state["depto_sel"]  = match_row['DEPARTAMENTO'].iloc[0]
                st.session_state["key_sel"]    = clicked_key
                st.rerun()

    st.caption(
        f"Valor {estadistico_mapa.lower()} departamental de intensidad de sequía acumulada — {periodo_mapa}. "
    )
else:
    st.info("ℹ️ Subí el archivo 'Departamentos_area_estudio_v3.geojson' al repositorio para ver el mapa.")

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtrado de Datos")

prov_sel = st.sidebar.selectbox("Provincia", provincias, key="prov_sel")
prov_norm = normalizar_nombre(prov_sel)

df_med_prov = df_med[df_med['_prov_norm'] == prov_norm]
deptos = sorted(df_med_prov['DEPARTAMENTO'].dropna().unique())

depto_sel = st.sidebar.selectbox("Departamento", deptos, key="depto_sel")
depto_norm = normalizar_nombre(depto_sel)

key_filtro = construir_key(prov_sel, depto_sel)

df_eme_depto = df_eme[df_eme['_key'] == key_filtro]
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
# DATOS DE SEQUÍA
# ─────────────────────────────────────────────────────────────────────
fila_med = df_med[df_med['_key'] == key_filtro]
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

fila_max = df_max[df_max['_key'] == key_filtro]
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
            f"⚠️ No se encontraron actividades para **{depto_sel}** en la base de emergencia."
        )
    else:
        st.info("ℹ️ No hay resoluciones de emergencia para esta selección.")

# ─────────────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSIÓN DEL DEPARTAMENTO SELECCIONADO
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Evaluación de la respuesta estatal — Matriz de confusión")
st.caption(
    "Compara los meses con evento de sequía (intensidad acumulada ≥ 6) contra los meses "
    "con resolución de emergencia declarada (mismo mes o hasta 1 mes después)."
)

# Selector del estadístico para la matriz
estadistico_mc = st.radio(
    "Estadístico para la matriz:",
    ["Mediana", "Máximo"],
    horizontal=True,
    key="estadistico_mc",
)

resultado_mc_depto = obtener_matriz_depto(key_filtro, estadistico_mc)

if resultado_mc_depto:
    VP = resultado_mc_depto['VP']
    FN = resultado_mc_depto['FN']
    FP = resultado_mc_depto['FP']
    VN = resultado_mc_depto['VN']

    col_mc, col_metricas = st.columns([1.3, 1])

    with col_mc:
        fig_mc = matriz_confusion_plotly(
            VP, FN, FP, VN,
            titulo=f"{depto_sel} ({prov_sel}) — {estadistico_mc}",
            height=460,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_metricas:
        metricas = calcular_metricas(VP, FN, FP, VN)

        def fmt_pct(x):
            return f"{x:.1%}" if pd.notna(x) else "—"

        st.markdown("**Métricas de desempeño**")
        st.metric("Exactitud (Accuracy)", fmt_pct(metricas['EXACTITUD']),
                  help="(VP+VN) / Total — % de meses clasificados correctamente")
        st.metric("Precisión (Precision)", fmt_pct(metricas['PRECISIÓN']),
                  help="VP / (VP+FP) — cuando se declaró emergencia, ¿había sequía?")
        st.metric("Sensibilidad (Recall)", fmt_pct(metricas['SENSIBILIDAD']),
                  help="VP / (VP+FN) — cuando hubo sequía, ¿se declaró emergencia?")
        st.metric("Especificidad (Specificity)", fmt_pct(metricas['ESPECIFICIDAD']),
                  help="VN / (VN+FP) — cuando NO hubo sequía, ¿NO se declaró emergencia?")

    # ─── TIEMPO DE RESPUESTA ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⏱️ Tiempo de respuesta del Estado")

    df_src = df_med if estadistico_mc == "Mediana" else df_max
    tiempo = calcular_tiempo_respuesta(
        key_filtro, df_src, df_eme, periodos_med, periodos_eme,
        umbral=6, ventana_max=6
    )

    if tiempo and tiempo['eventos_totales'] > 0:
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            if tiempo['promedio'] is not None:
                st.metric("Tiempo medio", f"{tiempo['promedio']:.1f} meses")
            else:
                st.metric("Tiempo medio", "—")
        with col_t2:
            if tiempo['minimo'] is not None:
                st.metric("Tiempo mínimo", f"{tiempo['minimo']} meses")
            else:
                st.metric("Tiempo mínimo", "—")
        with col_t3:
            if tiempo['maximo'] is not None:
                st.metric("Tiempo máximo", f"{tiempo['maximo']} meses")
            else:
                st.metric("Tiempo máximo", "—")
        with col_t4:
            st.metric("Eventos respondidos",
                      f"{tiempo['eventos_respondidos']} de {tiempo['eventos_totales']}")

        # Explicación
        if tiempo['tiempos']:
            delays_str = ", ".join([str(t) for t in sorted(tiempo['tiempos'])])
            st.caption(
                f"Delay (en meses) entre el inicio de cada evento y la primera "
                f"resolución declarada: **{delays_str}**"
            )
        else:
            st.warning("⚠️ Ningún evento de sequía tuvo respuesta estatal dentro de los 6 meses posteriores.")
    else:
        st.info("ℹ️ No se detectaron eventos de sequía en este departamento para calcular el tiempo de respuesta.")

    with st.expander("ℹ️ ¿Cómo interpretar esta matriz?"):
        total = VP + FN + FP + VN
        st.markdown(f"""
        **Departamento analizado:** {depto_sel} ({prov_sel})  
        **Estadístico:** {estadistico_mc}  
        **Umbral de sequía:** 6 (valor acumulado trimestral)  
        **Ventana de respuesta:** mismo mes o hasta 1 mes después  
        **Total de meses evaluados:** {total}

        | Categoría | Significado | Meses |
        |-----------|-------------|-------|
        | ✅ **VP** (Verdaderos Positivos) | Hubo evento de sequía Y resolución | {VP} |
        | ❌ **FN** (Falsos Negativos) | Hubo sequía pero NO se declaró emergencia | {FN} |
        | ⚠️ **FP** (Falsos Positivos) | Se declaró emergencia sin evento de sequía | {FP} |
        | · **VN** (Verdaderos Negativos) | Ni evento ni resolución | {VN} |

        **Métricas:**
        - **Exactitud** = (VP+VN) / Total
        - **Precisión** = VP / (VP+FP)
        - **Sensibilidad** = VP / (VP+FN)
        - **Especificidad** = VN / (VN+FP)

        **Tiempo de respuesta:** delay entre el inicio del evento y la primera resolución (en meses).
        """)
else:
    st.info("ℹ️ No hay datos suficientes para calcular la matriz en este departamento.")

# ─────────────────────────────────────────────────────────────────────
# ANÁLISIS GLOBAL POR PROVINCIA
# ─────────────────────────────────────────────────────────────────────
with st.expander("📊 Análisis global por provincia (clic para abrir)", expanded=False):
    st.markdown(
        "Matrices de confusión agregadas por provincia. "
        "**Umbral de sequía = 6**, **ventana de respuesta = 1 mes**."
    )

    estadistico_global = st.radio(
        "Estadístico para el análisis global:",
        ["Mediana", "Máximo"],
        horizontal=True,
        key="estadistico_global",
    )

    @st.cache_data
    def calcular_matrices_provincias(_estadistico):
        """Devuelve las matrices por provincia (usa precalculadas si están disponibles)."""
        if _estadistico == "Mediana" and df_mc_depto_med is not None:
            df = df_mc_depto_med.copy()
        elif _estadistico == "Máximo" and df_mc_depto_max is not None:
            df = df_mc_depto_max.copy()
        else:
            df_src = df_med if _estadistico == "Mediana" else df_max
            resultados = []
            for key in df_src['_key'].unique():
                r = matriz_confusion_depto(key, df_src, df_eme,
                                           periodos_med, periodos_eme,
                                           umbral=6, delay=1)
                if r:
                    resultados.append(r)
            df = pd.DataFrame([{
                'key': r['key'],
                'PROVINCIA': r['provincia'],
                'DEPARTAMENTO': r['departamento'],
                'VP': r['VP'], 'FN': r['FN'], 'FP': r['FP'], 'VN': r['VN'],
            } for r in resultados])

        df_prov = df.groupby('PROVINCIA').agg(
            N_DEPARTAMENTOS=('DEPARTAMENTO', 'nunique'),
            VP=('VP', 'sum'), FN=('FN', 'sum'),
            FP=('FP', 'sum'), VN=('VN', 'sum'),
        ).reset_index()

        def calc(r):
            m = calcular_metricas(r['VP'], r['FN'], r['FP'], r['VN'])
            return pd.Series({
                'EXACTITUD': m['EXACTITUD'],
                'PRECISIÓN': m['PRECISIÓN'],
                'SENSIBILIDAD': m['SENSIBILIDAD'],
                'ESPECIFICIDAD': m['ESPECIFICIDAD'],
            })

        df_prov = pd.concat([df_prov, df_prov.apply(calc, axis=1)], axis=1)
        df_prov = df_prov.sort_values('SENSIBILIDAD', ascending=False,
                                      na_position='last').reset_index(drop=True)
        return df_prov

    df_prov = calcular_matrices_provincias(estadistico_global)

    # ─── Tabla resumen ─────────────────────────────────────────────
    st.markdown("### Resumen por provincia")
    df_show = df_prov.copy()
    for col in ['EXACTITUD', 'PRECISIÓN', 'SENSIBILIDAD', 'ESPECIFICIDAD']:
        df_show[col] = df_show[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # ─── Grilla de matrices por provincia ──────────────────────────
    st.markdown("### Matrices de confusión por provincia")
    n = len(df_prov)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    for r in range(nrows):
        cols = st.columns(ncols)
        for c in range(ncols):
            idx = r * ncols + c
            if idx < n:
                row = df_prov.iloc[idx]
                with cols[c]:
                    fig_p = matriz_confusion_plotly(
                        int(row['VP']), int(row['FN']),
                        int(row['FP']), int(row['VN']),
                        titulo=f"{row['PROVINCIA']} (n={row['N_DEPARTAMENTOS']})",
                        height=420,
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

    # ─── Matriz global ─────────────────────────────────────────────
    st.markdown("### Matriz global (todas las provincias)")
    VP_g = int(df_prov['VP'].sum())
    FN_g = int(df_prov['FN'].sum())
    FP_g = int(df_prov['FP'].sum())
    VN_g = int(df_prov['VN'].sum())

    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        fig_global = matriz_confusion_plotly(
            VP_g, FN_g, FP_g, VN_g,
            titulo="Matriz global — todas las provincias",
            height=460,
        )
        st.plotly_chart(fig_global, use_container_width=True)

    with col_g2:
        st.markdown("**Métricas globales**")
        m_g = calcular_metricas(VP_g, FN_g, FP_g, VN_g)
        st.metric("Total departamentos", df_prov['N_DEPARTAMENTOS'].sum())
        st.metric("Exactitud",     f"{m_g['EXACTITUD']:.1%}"     if pd.notna(m_g['EXACTITUD'])     else "—")
        st.metric("Precisión",     f"{m_g['PRECISIÓN']:.1%}"     if pd.notna(m_g['PRECISIÓN'])     else "—")
        st.metric("Sensibilidad",  f"{m_g['SENSIBILIDAD']:.1%}"  if pd.notna(m_g['SENSIBILIDAD'])  else "—")
        st.metric("Especificidad", f"{m_g['ESPECIFICIDAD']:.1%}" if pd.notna(m_g['ESPECIFICIDAD']) else "—")

