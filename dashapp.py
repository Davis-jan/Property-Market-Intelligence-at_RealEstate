from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("cleaned_manageprop.csv")

df['date_sold'] = pd.to_datetime(df['date_sold'], errors='coerce', dayfirst=True)
df['year'] = df['date_sold'].dt.year

# =========================
# CREATE VALUATION PROXY
# =========================
df['valuation_price'] = (
    df.groupby(['location', 'property_type'])['price']
      .transform('mean')
)

# =========================
# APP INITIALISATION
# =========================
app = Dash(__name__)
server = app.server
app.title = "Property Market Intelligence"

# =========================
# STYLING
# =========================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Knight Frank Kenya | Property Intelligence</title>
    {%favicon%}
    {%css%}
    <style>
        body { font-family: Segoe UI, Arial; background:#f5f7fa; }
        .card {
            background:white;
            border-radius:10px;
            padding:20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        }
        .kpi-card {
            background:#002A5C;
            color:white;
            border-radius:10px;
            padding:20px;
            text-align:center;
        }
        .kpi-card h3 { margin:0; font-size:14px; opacity:0.8; }
        .kpi-card h1 { margin-top:10px; font-size:28px; }
        .section-title { margin-top:40px; color:#002A5C; }
        .credit { text-align:center; font-size:12px; color:#777; margin-top:-10px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

# =========================
# LAYOUT 
# =========================
app.layout = html.Div(style={'padding': '30px'}, children=[

    html.H1("Property Market Intelligence Dashboard",
            style={'textAlign': 'center', 'color': '#002A5C'}),

    html.Div("@RoundData Domain Limited", className="credit"),

    html.P("Executive analytics covering pricing, risk, forecasting, and valuation.",
           style={'textAlign': 'center', 'color': '#555'}),

    html.Div(className="card", children=[
        html.Label("Property Type"),
        dcc.Dropdown(
            df['property_type'].unique(),
            df['property_type'].unique()[0],
            id='property-type',
            clearable=False
        ),

        html.Br(),

        html.Label("Year Range"),
        dcc.RangeSlider(
            min=int(df['year'].min()),
            max=int(df['year'].max()),
            value=[int(df['year'].min()), int(df['year'].max())],
            marks={int(y): str(y) for y in range(int(df['year'].min()), int(df['year'].max()) + 1, 2)},
            id='year-slider'
        )
    ]),

    html.Div(
        id="kpi-row",
        style={'display':'grid','gridTemplateColumns':'repeat(4,1fr)',
               'gap':'20px','marginTop':'25px'}
    ),

    html.H2("Investment Risk Overview", className="section-title"),

    html.Div(
        style={'display':'grid','gridTemplateColumns':'1fr 2fr','gap':'20px'},
        children=[
            html.Div(className="card", children=[
                html.H4("Risk Score"),
                html.Div(id="risk-score-box",
                         style={'fontSize':'30px','fontWeight':'bold'})
            ]),
            html.Div(className="card", children=[dcc.Graph(id="risk-trend")])
        ]
    ),

    html.Div(
        style={'display':'grid','gridTemplateColumns':'1fr 1fr',
               'gap':'20px','marginTop':'30px'},
        children=[
            html.Div(className="card", children=[dcc.Graph(id="value-location")]),
            html.Div(className="card", children=[dcc.Graph(id="market-share")])
        ]
    ),

    html.H2("Valuation & Outlook", className="section-title"),

    html.Div(
        style={'display':'grid','gridTemplateColumns':'2fr 1fr','gap':'20px'},
        children=[
            html.Div(className="card", children=[dcc.Graph(id="valuation-gap")]),
            html.Div(className="card", children=[dcc.Graph(id="forecast")])
        ]
    ),

    html.Div(
        style={'display':'grid','gridTemplateColumns':'1fr 1fr',
               'gap':'20px','marginTop':'30px'},
        children=[
            html.Div(className="card", children=[
                html.H4("Overpriced Alerts"),
                html.Pre(id="overpriced-table")
            ]),
            html.Div(className="card", children=[
                html.H4("Executive Summary"),
                html.Pre(id="insights-text")
            ])
        ]
    )
])

# =========================
# CALLBACK
# =========================
@app.callback(
    Output("kpi-row","children"),
    Output("risk-score-box","children"),
    Output("risk-trend","figure"),
    Output("value-location","figure"),
    Output("market-share","figure"),
    Output("valuation-gap","figure"),
    Output("forecast","figure"),
    Output("overpriced-table","children"),
    Output("insights-text","children"),
    Input("property-type","value"),
    Input("year-slider","value")
)
def update_dashboard(property_type, year_range):

    dff = df[
        (df['property_type'] == property_type) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]

    if dff.empty:
        empty_fig = go.Figure()
        return [], "N/A", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "No data available.", "No insights available."

    # ================= KPI =================
    total_value = dff['price'].sum()
    avg_price = dff['price'].mean()
    transactions = len(dff)

    yearly = dff.groupby('year')['price'].sum()

    yoy = yearly.pct_change().iloc[-1]*100 if len(yearly) > 1 else 0

    kpis = [
        html.Div(className="kpi-card", children=[html.H3("Total Value"), html.H1(f"KES {total_value:,.0f}")]),
        html.Div(className="kpi-card", children=[html.H3("Avg Price"), html.H1(f"KES {avg_price:,.0f}")]),
        html.Div(className="kpi-card", children=[html.H3("Transactions"), html.H1(transactions)]),
        html.Div(className="kpi-card", children=[html.H3("YoY Growth"), html.H1(f"{yoy:.1f}%")])
    ]

    # ================= RISK =================
    volatility = dff['price'].std() / dff['price'].mean() * 100
    gap = abs(dff['price'] - dff['valuation_price']) / dff['valuation_price']

    risk_score = min((0.6*volatility + 0.4*gap.mean()*100), 100)
    risk_label = "Low" if risk_score < 35 else "Moderate" if risk_score < 65 else "High"

    risk_trend = (
        dff.groupby('year')['price'].std()
        / dff.groupby('year')['price'].mean() * 100
    ).reset_index(name="risk")

    fig_risk = px.line(risk_trend, x='year', y='risk', markers=True,
                       title="Risk Trend (Volatility-Based)")

    # ================= LOCATION =================
    loc = dff.groupby('location')['price'].sum().reset_index()
    fig_value = px.bar(loc, x='location', y='price')
    fig_share = px.pie(loc, names='location', values='price')

    # ================= VALUATION GAP =================
    fig_gap = px.scatter(dff, x='valuation_price', y='price')

    # ================= FORECAST =================
    fig_fc = go.Figure()

    if len(yearly) >= 4:
        ts = yearly.copy()
        ts.index = pd.to_datetime(ts.index, format="%Y")

        model = sm.tsa.ARIMA(ts, order=(1,1,1))
        res = model.fit()

        fc = res.get_forecast(steps=3)
        mean = fc.predicted_mean
        conf = fc.conf_int()

        fig_fc.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical"))
        fig_fc.add_trace(go.Scatter(x=mean.index, y=mean.values, name="Base Forecast"))

        fig_fc.add_trace(go.Scatter(
            x=conf.index, y=conf.iloc[:,0],
            fill='none', line=dict(width=0), showlegend=False
        ))
        fig_fc.add_trace(go.Scatter(
            x=conf.index, y=conf.iloc[:,1],
            fill='tonexty', name='Confidence Band'
        ))

        fig_fc.add_trace(go.Scatter(x=mean.index, y=mean.values * 1.1, name="Upside"))
        fig_fc.add_trace(go.Scatter(x=mean.index, y=mean.values * 0.9, name="Downside"))

        fig_fc.update_layout(title="ARIMA Forecast (Scenario-Based)")

    # ================= OVERPRICED =================
    overpriced = dff[dff['price'] > 1.2*dff['valuation_price']]
    alert = "No overpriced transactions." if overpriced.empty else overpriced[['location','price','valuation_price']].to_string(index=False)

    # ================= EXEC SUMMARY =================
    insights = f"""
Between {year_range[0]} and {year_range[1]}, {property_type} assets recorded
KES {total_value:,.0f} in total value across {transactions} transactions.

Market risk is classified as {risk_label.upper()}, driven by price volatility
and valuation dispersion.

Forecast scenarios indicate sensitivity to downside shocks,
with confidence intervals highlighting pricing uncertainty.
"""

    return kpis, f"{risk_score:.0f}/100 ({risk_label})", fig_risk, fig_value, fig_share, fig_gap, fig_fc, alert, insights

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False)

