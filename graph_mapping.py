from typing import List, Dict
from collections import Counter



# Keys are (col_count, type_signature_sorted_tuple)
GRAPH_RULES: Dict[tuple, List[str]] = {

    # Univariate
    (1, ("numerical",)): [
        "histogram",
        "box plot",
        "violin plot",
        "density plot",
    ],
    (1, ("categorical",)): [
        "bar chart",
        "pie chart",
    ],

    # Bivariate
    (2, ("numerical", "numerical")): [
        "scatter plot",
        "hexbin plot",
    ],
    (2, ("categorical", "numerical")): [
        "grouped box plot",
        "grouped violin plot",
        "aggregated bar chart",
    ],
    (2, ("datetime", "numerical")): [
        "line plot",
        "area plot",
    ],

    # Multivariate
    (3, ("numerical", "numerical", "numerical")): [
        "pair plot",
        "correlation heatmap",
    ],
}




# PUBLIC API

def get_graph_suggestions(
    selected_columns: List[str],
    column_metadata: List[dict]
) -> Dict[str, object]:
    """
    Returns ALL valid graph options for the selected columns.

    Parameters
    ----------
    selected_columns : list[str]
        Columns selected by the user (ANY number)

    column_metadata : list[dict]
        Metadata from ColumnSchema (name + detected_type)

    Returns
    -------
    dict with:
        - analysis_type
        - column_types
        - suggested_graphs
    """

    # Build column -> type mapping
    col_type_map = {
        col["name"]: col["detected_type"]
        for col in column_metadata
    }

    # Get detected types for selected columns
    selected_types = [
        col_type_map[col]
        for col in selected_columns
        if col in col_type_map
    ]

    col_count = len(selected_types)

    # Normalize for matching rules
    type_signature = tuple(sorted(selected_types))

    lookup_key = (col_count, type_signature)
    suggested_graphs = list(GRAPH_RULES.get(lookup_key, []))

    # For 3+ numeric columns fall back to the 3-numeric rule
    if not suggested_graphs and col_count >= 3:
        all_numeric = all(t == "numerical" for t in selected_types)
        if all_numeric:
            suggested_graphs = list(
                GRAPH_RULES.get((3, ("numerical", "numerical", "numerical")), [])
            )

    # Determine analysis type 
    if col_count == 1:
        analysis_type = "univariate"
    elif col_count == 2:
        analysis_type = "bivariate"
    else:
        analysis_type = "multivariate"

    return {
        "analysis_type": analysis_type,
        "column_types": dict(Counter(selected_types)),
        "suggested_graphs": sorted(set(suggested_graphs))
    }
"""from graph_mapping import get_graph_suggestions

columns = ["price", "rating"]
metadata = [
    {"name": "price", "detected_type": "numerical"},
    {"name": "rating", "detected_type": "numerical"},
    {"name": "city", "detected_type": "categorical"}
]

print(get_graph_suggestions(columns, metadata))
"""