import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 📥 Load PCA data
pca_df = pd.read_csv("data/pca_dashboard.csv")

# 📊 Auto metrics from hybrid model
auto_metrics = {
    "Accuracy": 0.86,
    "Precision": 0.88,
    "Recall": 0.80,
    "F1 Score": 0.84,
    "ROC AUC": 0.89
}

# 🧮 Manual metrics from confusion matrix
TP, TN, FP, FN = 16, 27, 4, 7
manual_metrics = {
    "Accuracy": (TP + TN) / (TP + TN + FP + FN),
    "Precision": TP / (TP + FP),
    "Recall": TP / (TP + FN),
    "F1 Score": 2 * ((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN)))
}

# 🧩 Confusion matrix
conf_matrix = np.array([[TN, FP], [FN, TP]])

# 🚀 Create Dash app
app = dash.Dash(__name__)
app.title = "Hybrid Mental Health Dashboard"

app.layout = html.Div([
    html.H1("🧠 Hybrid Mental Health Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.H3("📊 Clinical Metrics (Auto vs Manual)"),
        html.Div([
            html.Div([
                html.H4(metric),
                html.P(f"Auto: {auto_metrics[metric]:.2f}"),
                html.P(f"Manual: {manual_metrics[metric]:.2f}")
            ], style={
                "display": "inline-block",
                "margin": "10px",
                "padding": "10px",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "width": "180px",
                "textAlign": "center"
            })
            for metric in manual_metrics
        ])
    ], style={"textAlign": "center"}),

    html.Div([
        html.H3("🧮 Manual Calculation Steps"),
        html.Ul([
            html.Li("Accuracy = (TP + TN) / (TP + TN + FP + FN) = (16 + 27) / 54 = 0.80"),
            html.Li("Precision = TP / (TP + FP) = 16 / (16 + 4) = 0.80"),
            html.Li("Recall = TP / (TP + FN) = 16 / (16 + 7) = 0.70"),
            html.Li("F1 Score = 2 * (Precision * Recall) / (Precision + Recall) ≈ 0.75")
        ])
    ], style={"marginLeft": "40px"}),

    html.Div([
        html.H3("🧩 Confusion Matrix"),
        dcc.Graph(figure=go.Figure(
            data=go.Heatmap(
                z=conf_matrix,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale="Blues",
                showscale=True
            ),
            layout_title_text="Confusion Matrix"
        ))
    ]),

    html.Div([
        html.H3("📉 PCA Scatter Plot (2D)"),
        dcc.Graph(figure=px.scatter(
            pca_df, x="PC1", y="PC2",
            color=pca_df["Diagnosis"].map({0: "No Disorder", 1: "Disorder"}),
            hover_data=["PHQ", "GAD", "Epworth"],
            title="PCA Projection of Mental Health Data"
        ))
    ])
])

if __name__ == "__main__":
    app.run(debug=True)