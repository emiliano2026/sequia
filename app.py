import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

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
    nombre = str(nombre).strip()
    for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ñ','n'),
                 ('Á','A'),('É','E'),('Í','I'),('Ó','O'),('Ú','U'),('Ñ','N')]:
        nombre = nombre.replace(a, b)
    return re.sub(r'[^a-z0-9]', '', nombre.lower())

def buscar_columna(df, patrones):
    """Busca una columna por coincidencia flexible (sin tildes, sin mayúsculas)."""
    cols_norm = {normalizar_nombre(c): c for c in df.columns}
    for p in patrones:
        pn = normalizar_nombre(p)
        for cn, cr in cols_norm.items():
            if pn in cn or cn in pn:
                return cr
    return None

# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    URL_SEQ = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_sequia_mediana_v2.csv"
    URL_EME = "https://raw.githubusercontent.com/emiliano2026/sequia/main/BBDD_todo.csv"

    # ── Base de sequía (delimitador coma) ───────────────────────────
    df_seq = pd.read_csv(URL_SEQ, sep=',', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
    df_seq.columns = df_seq.columns.str.strip()

    # Columnas de meses: terminan en "_median"
    cols_seq = [c for c in df_seq.columns if c.endswith('_median')]
    mapa_seq = {}
    for c in cols_seq:
        m = re.match(r'([A-Z]{3})(\d{4})_median', c)
        if m:
            mapa_seq[c] = f"{m.group(1)}_{m.group(2)}"
    df_seq = df_seq.rename(columns=mapa_seq)
    periodos_seq = sorted(mapa_seq.values(),
                          key=lambda x: (int(x.split('_')[1]), x.split('_')[0]))

    for p in periodos_seq:
        df_seq[p] = pd.to_numeric(df_seq[p], errors='coerce')

    # Detección flexible de PROVINCIA y DEPARTAMENTO
    col_prov_seq = buscar_columna(df_seq, ['provincia'])
    col_dept_seq = buscar_columna(df_seq, ['departamento', 'nam', 'partido', 'municipio'])

    if not col_prov_seq or not col_dept_seq:
        st.error("❌ No se detectaron columnas de PROVINCIA/DEPARTAMENTO en la base de sequía.")
        st.write("**Columnas disponibles:**", list(df_seq.columns))
        st.stop()

    df_seq['PROVINCIA']    = df_seq[col_prov_seq].astype(str).str.strip().str.upper()
    df_seq['DEPARTAMENTO'] = df_seq[col_dept_seq].astype(str).str.strip().str.upper()

    # ── Base de emergencia (delimitador |) ──────────────────────────
    df_eme = pd.read_csv(URL_EME, sep='|', skipinitialspace=True,
                         dtype=str, encoding='utf-8-sig')
    df_eme.columns = df_eme.columns.str.strip()

    # Columnas de meses: formato MES_AAAA
    cols_eme = [c for c in df_eme.columns if re.match(r'^[A-Z]{3}_\d{4}$', c)]
    periodos_eme = sorted(cols_eme,
                          key=lambda x: (int(x.split('_')[1]), x.split('_')[0]))

    # Detección flexible
    col_prov_eme = buscar_columna(df_eme, ['provincia'])
    col_dept_eme = buscar_columna(df_eme, ['departamento', 'nam', 'partido', 'municipio'])
    col_act_eme  = buscar_columna(df_eme, ['actividad'])

    if not col_prov_eme or not col_dept_eme or not col_act_eme:
        st.error("❌ No se detectaron todas las columnas necesarias en BBDD_todo.")
        st.write("**Columnas detectadas en BBDD_todo:**", list(df_eme.columns))
        st.write("¿Se encontró PROVINCIA?", col_prov_eme)
        st.write("¿Se encontró DEPARTAMENTO/nam?", col_dept_eme)
        st.write("¿Se encontró ACTIVIDAD?", col_act_eme)
        st.stop()

    df_eme['PROVINCIA']    = df_eme[col_prov_eme].astype(str).str.strip().str.upper()
    df_eme['DEPARTAMENTO'] = df_eme[col_dept_eme].astype(str).str.strip().str.upper()
    df_eme['ACTIVIDAD']    = df_eme[col_act_eme].astype(str).str.strip()

    # ── Unión de períodos ───────────────────────────────────────────
    periodos_todos = sorted(set(periodos_seq) | set(periodos_eme),
                            key=lambda x: (int(x.split('_')[1]), x.split('_')[0]))

    return df_seq, df_eme, periodos_seq, periodos_eme, periodos_todos

df_seq, df_eme, periodos_seq, periodos_eme, periodos_todos = load_data()

# ─────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtros")

# Provincias (de la base de sequía)
provincias = sorted(df_seq['PROVINCIA'].dropna().unique())
prov_sel = st.sidebar.selectbox("📍 Provincia", provincias)

# Departamentos de esa provincia
df_seq_prov = df_seq[df_seq['PROVINCIA'] == prov_sel]
deptos = sorted(df_seq_prov['DEPARTAMENTO'].dropna().unique())
depto_sel = st.sidebar.selectbox("🏘️ Departamento", deptos)

# Actividades (opcional, de la base de emergencia)
df_eme_depto = df_eme[
    (df_eme['PROVINCIA'] == prov_sel) &
    (df_eme['DEPARTAMENTO'] == depto_sel)
]
actividades = sorted(df_eme_depto['ACTIVIDAD'].dropna().unique())
act_sel = st.sidebar.multiselect(
    "🌾 Actividades (opcional)",
    actividades,
    default=actividades
)

# ─────────────────────────────────────────────────────────────────────
# DATOS DE SEQUÍA (extendidos al eje completo)
# ─────────────────────────────────────────────────────────────────────
fila_seq = df_seq_prov[df_seq_prov['DEPARTAMENTO'] == depto_sel]
if fila_seq.empty:
    st.warning(f"⚠️ No hay datos de sequía para {depto_sel} ({prov_sel}).")
    st.stop()

# Crear serie de sequía alineada con periodos_todos
valores_seq = []
for p in periodos_todos:
    if p in periodos_seq:
        v = fila_seq[p].iloc[0]
        valores_seq.append(v if pd.notna(v) else np.nan)
    else:
        valores_seq.append(np.nan)  # mes sin dato de sequía

# ─────────────────────────────────────────────────────────────────────
# BANDAS DE EMERGENCIA (para todos los períodos)
# ─────────────────────────────────────────────────────────────────────
emergencias = {p: [] for p in periodos_todos}
df_eme_depto_filt = df_eme_depto[df_eme_depto['ACTIVIDAD'].isin(act_sel)] if act_sel else df_eme_depto

for _, row in df_eme_depto_filt.iterrows():
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

# Línea de intensidad
fig.add_trace(go.Scatter(
    x=periodos_todos,
    y=valores_seq,
    mode='lines+markers',
    name='Intensidad de sequía',
    line=dict(color='black', width=2.5),
    marker=dict(size=8, color='black'),
    connectgaps=False,
))

# Bandas verticales
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
    st.info("ℹ️ No hay resoluciones de emergencia para esta selección.")

# ─────────────────────────────────────────────────────────────────────
# EXPANDER DE DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Ver datos crudos (depuración)"):
    st.write("**Columnas BBDD_sequia:**", list(df_seq.columns))
    st.write("**Columnas BBDD_todo:**", list(df_eme.columns))
    st.write("**Períodos de sequía:**", periodos_seq)
    st.write("**Períodos de emergencia:**", periodos_eme)
    st.write("**Períodos totales (eje X):**", periodos_todos)
    st.write("**Valores de sequía (alineados):**", valores_seq)
    st.write("**Emergencias detectadas:**", {k: v for k, v in emergencias.items() if v})
