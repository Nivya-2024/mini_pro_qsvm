import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 📥 Load PCA data
pca_df = pd.read_csv("data/pca_dashboard.csv")

# 📊 Auto metrics from sklearn
auto_metrics = {
    "Accuracy": 0.72,
    "Precision": 0.71,
    "Recall": 0.57,
    "F1 Score": 0.63,
    "ROC AUC": 0.70
}

# 🧮 Manual metrics from confusion matrix
TP, TN, FP, FN = 12, 24, 5, 9
manual_metrics = {
    "Accuracy": (TP + TN) / (TP + TN + FP + FN),
    "Precision": TP / (TP + FP),
    "Recall": TP / (TP + FN),
    "F1 Score": 2 * ((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN)))
}

# 🧩 Confusion matrix
conf_matrix = np.array([[TN, FP], [FN, TP]])

# 🔍 Kernel matrix sample
kernel_matrix = np.array([
    [1.0, 0.18, 0.005, 0.14, 0.001],
    [0.18, 1.0, 0.06, 0.12, 0.08],
    [0.005, 0.06, 1.0, 0.002, 0.005],
    [0.14, 0.12, 0.002, 1.0, 0.36],
    [0.001, 0.08, 0.005, 0.36, 1.0]
])

# 🚀 Create Dash app
app = dash.Dash(__name__)
app.title = "QSVM Mental Health Dashboard"

app.layout = html.Div([
    html.H1("🧠 QSVM Mental Health Dashboard", style={"textAlign": "center"}),

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
            html.Li("Accuracy = (TP + TN) / (TP + TN + FP + FN) = (12 + 24) / 50 = 0.72"),
            html.Li("Precision = TP / (TP + FP) = 12 / (12 + 5) = 0.71"),
            html.Li("Recall = TP / (TP + FN) = 12 / (12 + 9) = 0.57"),
            html.Li("F1 Score = 2 * (Precision * Recall) / (Precision + Recall) ≈ 0.63")
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
        html.H3("🔍 Kernel Matrix Sample"),
        dcc.Graph(figure=go.Figure(
            data=go.Heatmap(
                z=kernel_matrix,
                colorscale="Viridis",
                showscale=True
            ),
            layout_title_text="Quantum Kernel Matrix"
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