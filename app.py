import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import unicodedata

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Sequía IIPAC", layout="wide")

try:
    st.sidebar.image("LogoIIPAC.jpg", use_container_width=True)
except:
    st.sidebar.warning("Logo no encontrado. Subí 'LogoIIPAC.jpg'.")

st.title("🌵 Dashboard de Sequía IIPAC")
st.caption("Intensidad de sequía y resoluciones de emergencia agropecuaria")

# ─────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────
def normalizar_nombre(nombre):
    """Normaliza un nombre: minúsculas, sin tildes, sin prefijos como 'partido de'."""
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
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    URL_SEQ = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_sequia_mediana_v2.csv"
    URL_EME = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_todo_V2.csv"

    # ── Base de sequía (delimitador coma) ───────────────────────────
    df_seq = pd.read_csv(URL_SEQ, sep=',', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
    df_seq.columns = df_seq.columns.str.strip()

    cols_seq = [c for c in df_seq.columns if c.endswith('_median')]
    mapa_seq = {}
    for c in cols_seq:
        m = re.match(r'([A-Z]{3})(\d{4})_median', c)
        if m:
            mapa_seq[c] = f"{m.group(1)}_{m.group(2)}"
    df_seq = df_seq.rename(columns=mapa_seq)

    # ✅ RESPETAR EL ORDEN ORIGINAL DEL CSV (ya está en orden cronológico)
    periodos_seq = list(mapa_seq.values())

    for p in periodos_seq:
        df_seq[p] = pd.to_numeric(df_seq[p], errors='coerce')

    col_prov_seq = buscar_columna(df_seq, ['provincia'])
    col_dept_seq = buscar_columna(df_seq, ['departamento', 'nam', 'partido', 'municipio'])

    if not col_prov_seq or not col_dept_seq:
        st.error("❌ No se detectaron columnas de PROVINCIA/DEPARTAMENTO en la base de sequía.")
        st.write("**Columnas disponibles:**", list(df_seq.columns))
        st.stop()

    df_seq['PROVINCIA']    = df_seq[col_prov_seq].astype(str).str.strip()
    df_seq['DEPARTAMENTO'] = df_seq[col_dept_seq].astype(str).str.strip()
    df_seq['_prov_norm']   = df_seq['PROVINCIA'].apply(normalizar_nombre)
    df_seq['_dept_norm']   = df_seq['DEPARTAMENTO'].apply(normalizar_nombre)

    # ── Base de emergencia (delimitador coma) ───────────────────────
    df_eme = pd.read_csv(URL_EME, sep=',', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
    df_eme.columns = df_eme.columns.str.strip()

    cols_eme = [c for c in df_eme.columns if re.match(r'^[A-Z]{3}_\d{4}$', c)]

    # ✅ RESPETAR EL ORDEN ORIGINAL DEL CSV
    periodos_eme = list(cols_eme)

    col_prov_eme = buscar_columna(df_eme, ['provincia'])
    col_dept_eme = buscar_columna(df_eme, ['departamento', 'nam', 'partido', 'municipio'])
    col_act_eme  = buscar_columna(df_eme, ['actividad'])

    if not col_prov_eme or not col_dept_eme or not col_act_eme:
        st.error("❌ No se detectaron todas las columnas necesarias en BBDD_todo_V2.")
        st.write("**Columnas detectadas:**", list(df_eme.columns))
        st.stop()

    df_eme['PROVINCIA']    = df_eme[col_prov_eme].astype(str).str.strip()
    df_eme['DEPARTAMENTO'] = df_eme[col_dept_eme].astype(str).str.strip()
    df_eme['ACTIVIDAD']    = df_eme[col_act_eme].astype(str).str.strip()
    df_eme['_prov_norm']   = df_eme['PROVINCIA'].apply(normalizar_nombre)
    df_eme['_dept_norm']   = df_eme['DEPARTAMENTO'].apply(normalizar_nombre)

    # ✅ UNIÓN DE PERÍODOS PRESERVANDO ORDEN CRONOLÓGICO
    # Primero los de sequía (en su orden), luego los de emergencia que no estén
    periodos_todos = list(periodos_seq)
    for p in periodos_eme:
        if p not in periodos_todos:
            periodos_todos.append(p)

    return df_seq, df_eme, periodos_seq, periodos_eme, periodos_todos

df_seq, df_eme, periodos_seq, periodos_eme, periodos_todos = load_data()

# ─────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtros")

provincias = sorted(df_seq['PROVINCIA'].dropna().unique())
prov_sel = st.sidebar.selectbox("📍 Provincia", provincias)
prov_norm = normalizar_nombre(prov_sel)

df_seq_prov = df_seq[df_seq['_prov_norm'] == prov_norm]
deptos = sorted(df_seq_prov['DEPARTAMENTO'].dropna().unique())
depto_sel = st.sidebar.selectbox("🏘️ Departamento", deptos)
depto_norm = normalizar_nombre(depto_sel)

# Cruce robusto
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
    "🌾 Actividades (opcional)",
    actividades,
    default=actividades if actividades else []
)

# ─────────────────────────────────────────────────────────────────────
# DATOS DE SEQUÍA
# ─────────────────────────────────────────────────────────────────────
fila_seq = df_seq_prov[df_seq_prov['DEPARTAMENTO'] == depto_sel]
if fila_seq.empty:
    st.warning(f"⚠️ No hay datos de sequía para {depto_sel} ({prov_sel}).")
    st.stop()

valores_seq = []
for p in periodos_todos:
    if p in periodos_seq:
        v = fila_seq[p].iloc[0]
        valores_seq.append(v if pd.notna(v) else np.nan)
    else:
        valores_seq.append(np.nan)

# ─────────────────────────────────────────────────────────────────────
# BANDAS DE EMERGENCIA
# ─────────────────────────────────────────────────────────────────────
emergencias = {p: [] for p in periodos_todos}
if act_sel and not df_eme_depto.empty:
    df_eme_filt = df_eme_depto[df_eme_depto['ACTIVIDAD'].isin(act_sel)]
    for _, row in df_eme_filt.iterrows():
        act = row['ACTIVIDAD']
        for p in periodos_todos:
            if p in periodos_eme:
                val = str(row[p]).strip()
                if val and val.lower() not in ('nan', 'none', '', 'null'):
                    emergencias[p].append((act, val))

# ─────────────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────────────
st.subheader(f"📈 Evolución de la sequía — {depto_sel} ({prov_sel})")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=periodos_todos,
    y=valores_seq,
    mode='lines+markers',
    name='Intensidad de sequía',
    line=dict(color='black', width=2.5),
    marker=dict(size=8, color='black'),
    connectgaps=False,
))

paleta = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#17becf', '#bcbd22']
mapa_colores = {act: paleta[i % len(paleta)] for i, act in enumerate(actividades)}

i = 0
while i < len(periodos_todos):
    p = periodos_todos[i]
    if emergencias[p]:
        acts_mes = set(a for a, _ in emergencias[p])
        j = i
        while j + 1 < len(periodos_todos) and emergencias[periodos_todos[j + 1]]:
            j += 1
        x0 = i - 0.5
        x1 = j + 0.5
        color = mapa_colores.get(list(acts_mes)[0], '#888888') if len(acts_mes) == 1 else '#888888'
        resoluciones = sorted(set(r for k in range(i, j + 1) for _, r in emergencias[periodos_todos[k]]))
        etiqueta = f"Res. {', '.join(resoluciones[:3])}" + ("..." if len(resoluciones) > 3 else "")
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color, opacity=0.18, line_width=0,
            annotation_text=etiqueta,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color=color),
        )
        i = j + 1
    else:
        i += 1

fig.update_layout(
    xaxis=dict(
        title="Mes",
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
    ),
    yaxis=dict(title="Intensidad de sequía (0-9)", range=[-0.5, 9.5], dtick=1),
    template='plotly_white',
    height=500,
    legend=dict(orientation='h', y=-0.25),
    margin=dict(l=40, r=40, t=40, b=80),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TABLA DE RESOLUCIONES
# ─────────────────────────────────────────────────────────────────────
st.subheader("📋 Resoluciones de emergencia declaradas")

filas = []
for p in periodos_todos:
    for act, res in emergencias[p]:
        filas.append({'Mes': p, 'Actividad': act, 'Resolución': res})

if filas:
    df_tabla = pd.DataFrame(filas)
    df_tabla['Intensidad'] = df_tabla['Mes'].apply(
        lambda m: valores_seq[periodos_todos.index(m)]
    )
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
    st.write("### Base de sequía")
    st.write("**Provincias disponibles (base sequía):**", sorted(df_seq['PROVINCIA'].unique())[:20])
    st.write(f"**Departamentos de '{prov_sel}' (base sequía):**",
             sorted(df_seq_prov['DEPARTAMENTO'].unique())[:30])

    st.write("### Base de emergencia")
    st.write("**Provincias disponibles (base emergencia):**",
             sorted(df_eme['PROVINCIA'].unique())[:20])
    st.write(f"**Departamentos de '{prov_sel}' (base emergencia):**",
             sorted(df_eme[df_eme['_prov_norm'] == prov_norm]['DEPARTAMENTO'].unique())[:30])

    st.write("### Cruce actual")
    st.write(f"Departamento seleccionado (sequía): `{depto_sel}`")
    st.write(f"Departamento normalizado: `{depto_norm}`")
    st.write(f"Filas encontradas en emergencia: **{len(df_eme_depto)}**")
    if len(df_eme_depto) > 0:
        st.write("Actividades detectadas:", sorted(df_eme_depto['ACTIVIDAD'].unique()))

    st.write("### Fechas")
    st.write(f"**Períodos sequía ({len(periodos_seq)}):**", periodos_seq)
    st.write(f"**Períodos emergencia ({len(periodos_eme)}):**", periodos_eme)
    st.write(f"**Períodos totales ({len(periodos_todos)}):**", periodos_todos)
