# ─────────────────────────────────────────────────────────────────────
# COLORES POR ACTIVIDAD
# ─────────────────────────────────────────────────────────────────────
COLORES_ACTIVIDAD = {
    'agriculturafamiliar': '#808080',  # gris
    'agricultura': '#90EE90',           # verde claro
    'ganaderia': '#8B4513',             # marrón
    'apicultura': '#FFD700',            # amarillo
    'fruticultura': '#FF0000',          # rojo
    'horticultura': '#FFA500',          # naranja
    'psicultura': '#87CEEB',            # celeste
    'piscicultura': '#87CEEB',          # celeste
    'silvicultura': '#006400',          # verde oscuro
}

def color_para_actividad(act):
    act_norm = normalizar_nombre(act)
    for key, color in COLORES_ACTIVIDAD.items():
        if key in act_norm:
            return color
    return '#cccccc'

# ─────────────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────────────
st.subheader(f"📈 Evolución de la sequía — {depto_sel} ({prov_sel})")

n_acts = len(act_sel)

if n_acts == 0:
    # Solo gráfico de sequía, sin subplots
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
    fig.add_hline(
        y=6,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
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
        yaxis=dict(title="Intensidad de sequía (0-9)", range=[-0.5, 9.5], dtick=1),
        template='plotly_white',
        height=500,
        margin=dict(l=40, r=40, t=40, b=80),
    )
else:
    # Con subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # Subplot 1: sequía
    fig.add_trace(go.Scatter(
        x=periodos_todos,
        y=valores_seq,
        mode='lines+markers',
        name='Intensidad de sequía',
        line=dict(color='black', width=2.5),
        marker=dict(size=8, color='black'),
        connectgaps=False,
        showlegend=True,
    ), row=1, col=1)

    fig.add_hline(
        y=6,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        annotation_text="Límite sequía (6)",
        annotation_position="top right",
        row=1, col=1,
    )

    # Subplot 2: franjas por actividad
    for idx, act in enumerate(act_sel):
        color = color_para_actividad(act)
        xs, ys, resoluciones = [], [], []
        for _, row in df_eme_depto.iterrows():
            if row['ACTIVIDAD'] != act:
                continue
            for p in periodos_todos:
                if p in periodos_eme:
                    val = str(row[p]).strip()
                    if val and val.lower() not in ('nan', 'none', '', 'null'):
                        xs.append(p)
                        ys.append(idx)
                        resoluciones.append(val)
        if xs:
            hover_text = [
                f"<b>{act}</b><br>Mes: {x}<br>Resolución: {r}"
                for x, r in zip(xs, resoluciones)
            ]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                name=act,
                marker=dict(
                    symbol='square',
                    size=22,
                    color=color,
                    line=dict(color='white', width=1),
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True,
            ), row=2, col=1)

    # Configurar ejes
    fig.update_xaxes(
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        row=1, col=1,
    )
    fig.update_xaxes(
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(periodos_todos))),
        ticktext=periodos_todos,
        range=[-0.5, len(periodos_todos) - 0.5],
        title="Mes",
        row=2, col=1,
    )
    fig.update_yaxes(
        title="Intensidad de sequía (0-9)",
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
        height=650,
        margin=dict(l=40, r=40, t=20, b=80),
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    )

st.plotly_chart(fig, use_container_width=True)
