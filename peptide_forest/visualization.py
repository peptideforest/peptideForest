"""Visualizing training progress and model performance."""
import pandas as pd
import plotly.express as px


def plot_model_performance(training_performance, title=None):
    """Plot model performance."""
    kpi_df = (
        pd.DataFrame(training_performance)
        .T.reset_index()
        .rename(
            columns={
                "index": "epoch",
                "mae": "Mean absolute error",
                "mse": "Mean squared error",
                "rmse": "Root mean squared error",
                "r2": "R2 score",
            }
        )
        .melt(id_vars=["epoch"], var_name="metric", value_name="value")
    )

    fig = (
        px.scatter(kpi_df, x="epoch", y="value", color="metric", range_y=[0, 1])
        .update_traces(mode="lines+markers")
        .update_layout(
            title="Model performance",
            xaxis_title="Epoch",
            yaxis_title="Metric value",
            legend_title="Metrics",
        )
    )
    if title is not None:
        fig.update_layout(title=title)

    fig.show()
    print()
