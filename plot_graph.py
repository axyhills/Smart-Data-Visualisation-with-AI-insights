import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_graph(df: pd.DataFrame, chart_type: str, columns: list):


    # UNIVARIATE

    if chart_type == "histogram":
        return px.histogram(df, x=columns[0])

    elif chart_type == "box plot":
        return px.box(df, y=columns[0])

    elif chart_type == "violin plot":
        return px.violin(df, y=columns[0], box=True)

    elif chart_type == "density plot":
        return px.histogram(df, x=columns[0], histnorm="probability density")

    elif chart_type == "bar chart":
        counts = df[columns[0]].value_counts().reset_index()
        counts.columns = [columns[0], "count"]
        return px.bar(counts, x=columns[0], y="count")

    elif chart_type == "pie chart":
        counts = df[columns[0]].value_counts().reset_index()
        counts.columns = [columns[0], "count"]
        return px.pie(counts, names=columns[0], values="count")


    # BIVARIATE

    elif chart_type == "scatter plot":
        return px.scatter(df, x=columns[0], y=columns[1])

    elif chart_type == "hexbin plot":
        return px.density_heatmap(df, x=columns[0], y=columns[1])

    elif chart_type == "grouped box plot":
        return px.box(df, x=columns[0], y=columns[1])

    elif chart_type == "grouped violin plot":
        return px.violin(df, x=columns[0], y=columns[1], box=True)

    elif chart_type == "aggregated bar chart":
        agg = df.groupby(columns[0])[columns[1]].mean().reset_index()
        return px.bar(agg, x=columns[0], y=columns[1])

    elif chart_type == "line plot":
        return px.line(df.sort_values(columns[0]), x=columns[0], y=columns[1])

    elif chart_type == "area plot":
        return px.area(df.sort_values(columns[0]), x=columns[0], y=columns[1])


    # MULTIVARIATE

    elif chart_type == "correlation heatmap":
        corr = df[columns].corr()
        return px.imshow(corr, text_auto=True)

    elif chart_type == "pair plot":
        return px.scatter_matrix(df, dimensions=columns)

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")


