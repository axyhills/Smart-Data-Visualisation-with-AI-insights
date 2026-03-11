from itertools import combinations
from typing import List, Dict, Tuple, Set
import logging
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
import config
from graph_mapping import get_graph_suggestions

warnings.filterwarnings('ignore')\

logger = logging.getLogger(__name__)



# UTILITY FUNCTIONS


def is_constant_column(df, col: str) -> bool:
    return df[col].nunique(dropna=True) <= 1


def get_missing_ratio(df, col: str) -> float:
    return df[col].isna().sum() / len(df)


def get_effective_cardinality(df, col: str, threshold: float = 0.01) -> int:
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[col].nunique(dropna=True)
    value_counts = df[col].value_counts(normalize=True, dropna=True)
    return (value_counts >= threshold).sum()


def is_high_cardinality_categorical(df, col: str, max_categories: int = 30) -> bool:
    if pd.api.types.is_numeric_dtype(df[col]):
        return False
    return df[col].nunique(dropna=True) > max_categories


def has_sufficient_variance(df, col: str, min_variance_ratio: float = 0.01) -> bool:
    if not pd.api.types.is_numeric_dtype(df[col]):
        return True
    col_data = df[col].dropna()
    if len(col_data) == 0:
        return False
    std = col_data.std()
    mean = col_data.mean()
    if abs(mean) < 1e-10:
        return std > 1e-10
    return abs(std / mean) > min_variance_ratio


def detect_outliers_iqr(df, col: str, threshold: float = 3.0) -> float:
    if not pd.api.types.is_numeric_dtype(df[col]):
        return 0.0
    col_data = df[col].dropna()
    if len(col_data) < 4:
        return 0.0
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return 0.0
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
    return outliers / len(col_data)



# STATISTICAL RELATIONSHIP TESTS


def test_numerical_numerical_relationship(
    df, col1, col2, min_correlation=0.1, min_pvalue=0.05
) -> Tuple[bool, float, str]:
    data = df[[col1, col2]].dropna()
    if len(data) < 10:
        return False, 0.0, "insufficient_data"
    try:
        corr, p_value = pearsonr(data[col1], data[col2])
        if abs(corr) >= min_correlation and p_value < min_pvalue:
            return True, abs(corr), "pearson"
    except Exception:
        pass
    try:
        corr, p_value = spearmanr(data[col1], data[col2])
        if abs(corr) >= min_correlation and p_value < min_pvalue:
            return True, abs(corr), "spearman"
    except Exception:
        pass
    return False, 0.0, "no_relationship"


def test_categorical_numerical_relationship(
    df, cat_col, num_col, min_pvalue=0.05, min_effect_size=0.1
) -> Tuple[bool, float, str]:
    data = df[[cat_col, num_col]].dropna()
    if len(data) < 10:
        return False, 0.0, "insufficient_data"
    categories = data[cat_col].unique()
    if len(categories) < 2:
        return False, 0.0, "insufficient_categories"
    groups = [data[data[cat_col] == cat][num_col].values for cat in categories]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return False, 0.0, "insufficient_samples"
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        if p_value < min_pvalue:
            grand_mean = data[num_col].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((data[num_col] - grand_mean) ** 2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            if eta_squared >= min_effect_size:
                return True, eta_squared, "anova"
    except Exception:
        pass
    return False, 0.0, "no_relationship"


def test_categorical_categorical_relationship(
    df, col1, col2, min_pvalue=0.05, min_cramers_v=0.1
) -> Tuple[bool, float, str]:
    data = df[[col1, col2]].dropna()
    if len(data) < 10:
        return False, 0.0, "insufficient_data"
    contingency_table = pd.crosstab(data[col1], data[col2])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return False, 0.0, "insufficient_variation"
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        if (expected < 5).sum() / expected.size > 0.2:
            return False, 0.0, "low_expected_frequencies"
        if p_value < min_pvalue:
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            if cramers_v >= min_cramers_v:
                return True, cramers_v, "chi_square"
    except Exception:
        pass
    return False, 0.0, "no_relationship"


def has_significant_relationship(
    df, col1, col2, relationship_config=None
) -> Tuple[bool, float, str]:
    if relationship_config is None:
        relationship_config = {
            "min_correlation": 0.15,
            "min_pvalue": 0.05,
            "min_effect_size": 0.1,
            "min_cramers_v": 0.15,
            "min_mi_score": 0.05,
        }
    is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
    is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])

    if is_col1_numeric and is_col2_numeric:
        return test_numerical_numerical_relationship(
            df, col1, col2,
            min_correlation=relationship_config["min_correlation"],
            min_pvalue=relationship_config["min_pvalue"],
        )
    elif is_col1_numeric or is_col2_numeric:
        cat_col = col2 if is_col1_numeric else col1
        num_col = col1 if is_col1_numeric else col2
        return test_categorical_numerical_relationship(
            df, cat_col, num_col,
            min_pvalue=relationship_config["min_pvalue"],
            min_effect_size=relationship_config["min_effect_size"],
        )
    else:
        return test_categorical_categorical_relationship(
            df, col1, col2,
            min_pvalue=relationship_config["min_pvalue"],
            min_cramers_v=relationship_config["min_cramers_v"],
        )



# VALIDITY FILTERS


def is_data_quality_sufficient(
    df, columns, max_missing_ratio=0.5, max_outlier_ratio=0.4
) -> Tuple[bool, str]:
    """
    Relaxed thresholds — real-world datasets (price, reviews, fees)
    naturally have high outlier ratios. Only reject truly broken columns.
    """
    for col in columns:
        if get_missing_ratio(df, col) > max_missing_ratio:
            return False, f"high_missing_values_{col}"
      
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            skewness = abs(col_data.skew()) if len(col_data) > 3 else 0
          
            if skewness < 2.0:
                if detect_outliers_iqr(df, col) > max_outlier_ratio:
                    return False, f"high_outliers_{col}"
    return True, "pass"


def is_real_life_valid(
    graph_type: str,
    columns: List[str],
    df: pd.DataFrame,
    dataset_schema,
    quality_config: Dict = None,
) -> Tuple[bool, str]:
    """
    Enhanced real-world graph validity rules.
    Returns (is_valid, reason).
    """
    if quality_config is None:
        quality_config = {
            "max_missing_ratio": 0.5,
            "max_outlier_ratio": 0.4,
            "min_variance_ratio": 0.01,
            "max_bar_categories": 20,
            "min_histogram_samples": 20,
            "min_heatmap_samples": 30,
            "min_scatter_samples": 15,
        }

    schema_map = {c.name: c for c in dataset_schema.columns}
    gt = graph_type.lower().strip()   
    # 1️⃣ Exclude ID and pure text columns
    for col in columns:
        if col not in schema_map:
            return False, f"unknown_column_{col}"
        if schema_map[col].detected_type in {"id", "text"}:
            return False, f"invalid_type_{col}"

    # 2️⃣ Constant columns
    for col in columns:
        if is_constant_column(df, col):
            return False, f"constant_column_{col}"

    # 3️⃣ Data quality
    is_quality_ok, quality_reason = is_data_quality_sufficient(
        df, columns,
        max_missing_ratio=quality_config["max_missing_ratio"],
        max_outlier_ratio=quality_config["max_outlier_ratio"],
    )
    if not is_quality_ok:
        return False, quality_reason

    # 4️⃣ Variance — only reject near-constant columns (std ~0)
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            if len(col_data) > 0 and col_data.std() < 1e-9:
                return False, f"low_variance_{col}"

    # 5️⃣ Graph-specific rules (all comparisons use normalised `gt`)

    if "bar" in gt:
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                if is_high_cardinality_categorical(df, col, quality_config["max_bar_categories"]):
                    return False, f"high_cardinality_{col}"

    if gt == "histogram":
        if len(df) < quality_config["min_histogram_samples"]:
            return False, "insufficient_samples"

    if gt == "correlation heatmap":
        if len(df) < quality_config["min_heatmap_samples"]:
            return False, "insufficient_samples_heatmap"

    if "scatter" in gt:
        if len(df) < quality_config["min_scatter_samples"]:
            return False, "insufficient_samples_scatter"

    # FIX: was checking "Area" (capital A) – now lowercase consistent
    if "line" in gt or "area" in gt:
        col = columns[0]
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, "not_temporal_data"

    if "pie" in gt:
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                if get_effective_cardinality(df, col) > 8:
                    return False, f"too_many_categories_pie_{col}"

    if gt == "density plot" and len(df) < 30:
        return False, "insufficient_samples_density"

    if gt == "hexbin plot" and len(df) < 100:
        return False, "insufficient_samples_hexbin"

    if gt == "pair plot":
        numeric_count = sum(pd.api.types.is_numeric_dtype(df[col]) for col in columns)
        if numeric_count > 5:
            return False, "too_many_dimensions_pairplot"

    return True, "pass"



# REDUNDANCY DETECTION


def is_redundant_graph(
    new_graph: Dict, existing_graphs: List[Dict], df: pd.DataFrame
) -> Tuple[bool, str]:
    new_cols = set(new_graph["columns"])
    new_type = new_graph["graph"].lower()

   
    similar_type_groups = [
        {"bar chart", "aggregated bar chart"},
        {"line plot", "area plot"},
        {"scatter plot", "hexbin plot"},
        {"histogram", "density plot"},
        {"box plot", "violin plot"},
        {"grouped box plot", "grouped violin plot"},
        {"grouped box plot", "aggregated bar chart"},
        {"grouped violin plot", "aggregated bar chart"},
    ]

    for existing in existing_graphs:
        existing_cols = set(existing["columns"])
        existing_type = existing["graph"].lower()

        # Exact duplicate
        if new_cols == existing_cols and new_type == existing_type:
            return True, "duplicate"

        # Similar graph types for the same column set
        if new_cols == existing_cols:
            for group in similar_type_groups:
                if new_type in group and existing_type in group:
                    return True, "similar_type"

        # Univariate subset of a bivariate with same graph type
        if (
            len(new_cols) == 1
            and new_cols.issubset(existing_cols)
            and existing.get("analysis_type") == "bivariate"
            and new_type == existing_type
        ):
            return True, "subset_redundant"

    return False, "not_redundant"



# GRAPH SCORING


def score_graph_importance(
    graph: Dict, df: pd.DataFrame, relationship_strength: float = 0.0
) -> float:
    score = 1.0
    columns = graph["columns"]
    graph_type = graph["graph"].lower()

    if len(columns) == 2:
        score += relationship_strength * 5.0

    if len(columns) == 1:
        col = columns[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = abs(df[col].skew())
            if 0.5 < skewness < 2.0:
                score += 0.5
        else:
            value_counts = df[col].value_counts(normalize=True)
            entropy = stats.entropy(value_counts)
            max_entropy = np.log(len(value_counts))
            if max_entropy > 0:
                score += (entropy / max_entropy) * 0.5

    preferred_types = {
        "correlation heatmap": 1.35,
        "pair plot": 1.30,
        "scatter plot": 1.25,
        "grouped box plot": 1.25,
        "grouped violin plot": 1.25,
        "line plot": 1.20,
        "area plot": 1.15,
        "hexbin plot": 1.10,
        "box plot": 1.10,
        "violin plot": 1.10,
        "histogram": 1.00,
        "aggregated bar chart": 1.00,
        "bar chart": 1.00,
        "density plot": 0.95,
        "pie chart": 0.70,
    }
    for pref_type, multiplier in preferred_types.items():
        if pref_type in graph_type:
            score *= multiplier
            break

    if len(columns) >= 3:
        score += 1.0

    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            score += 0.5
            break

    return score



# MAIN GRAPH GENERATOR


def generate_ranked_insights(
    df: pd.DataFrame,
    dataset_schema,
    precomputed_stats: Dict = None,
    require_relationships: bool = True,
    only_types: List[str] = None,
) -> List[Dict]:
    """
    Generates statistically valid, non-redundant, ranked graphs for the dataset.

    Parameters
    ----------
    only_types : list of analysis types to generate, e.g. ["univariate"].
                 None = generate all types (original behaviour).
    """
    if precomputed_stats is None:
        precomputed_stats = {}

    corr_matrix = precomputed_stats.get("corr_matrix")
    numeric_cols = precomputed_stats.get("numeric_cols", [])

    valid_graphs = []

    usable_columns = [col for col in dataset_schema.columns]
    metadata = [{"name": col.name, "detected_type": col.detected_type} for col in usable_columns]
    col_names = [col["name"] for col in metadata]

    logger.debug(f"Starting graph generation for {len(col_names)} columns…")

    if only_types is None or "univariate" in only_types:
        # UNIVARIATE
        logger.debug("Generating univariate graphs…")
        for col in col_names:
            graph_info = get_graph_suggestions([col], metadata)
            for graph in graph_info["suggested_graphs"]:
                is_valid, _ = is_real_life_valid(graph, [col], df, dataset_schema)
                if not is_valid:
                    continue
                graph_dict = {
                    "graph": graph,
                    "columns": [col],
                    "analysis_type": "univariate",
                    "relationship_strength": 0.0,
                    "relationship_method": "n/a",
                }
                is_redundant, _ = is_redundant_graph(graph_dict, valid_graphs, df)
                if not is_redundant:
                    graph_dict["score"] = score_graph_importance(graph_dict, df)
                    valid_graphs.append(graph_dict)

        logger.debug(f"  → {len(valid_graphs)} univariate graphs")


    if only_types is None or any(t in only_types for t in ("bivariate", "time_series")):
        # BIVARIATE
        logger.debug("Generating bivariate graphs…")
        bivariate_count = 0

        for c1, c2 in combinations(col_names, 2):
            is_num1 = c1 in numeric_cols
            is_num2 = c2 in numeric_cols

            
            if (
                is_num1 and is_num2
                and corr_matrix is not None
                and c1 in corr_matrix.columns
                and c2 in corr_matrix.columns
            ):
                strength = corr_matrix.loc[c1, c2]
                has_relationship = (not pd.isna(strength)) and (
                    abs(strength) >= 0.1  
                )
                method = "pearson_precomputed"
            else:
                has_relationship, strength, method = has_significant_relationship(df, c1, c2)

            if require_relationships and not has_relationship:
                continue

            graph_info = get_graph_suggestions([c1, c2], metadata)
            for graph in graph_info["suggested_graphs"]:
                is_valid, _ = is_real_life_valid(graph, [c1, c2], df, dataset_schema)
                if not is_valid:
                    continue
                graph_dict = {
                    "graph": graph,
                    "columns": [c1, c2],
                    "analysis_type": "bivariate",
                    "relationship_strength": float(strength) if not pd.isna(strength) else 0.0,
                    "relationship_method": method,
                }
                is_redundant, _ = is_redundant_graph(graph_dict, valid_graphs, df)
                if not is_redundant:
                    graph_dict["score"] = score_graph_importance(graph_dict, df, float(strength) if not pd.isna(strength) else 0.0)
                    valid_graphs.append(graph_dict)
                    bivariate_count += 1

        logger.debug(f"  → {bivariate_count} bivariate graphs")


    if only_types is None or "multivariate" in only_types:
        # MULTIVARIATE
        logger.debug("Generating multivariate graphs…")
        if not numeric_cols:
            numeric_cols = [
                col.name for col in dataset_schema.columns if col.detected_type == "numerical"
            ]

        
        if len(numeric_cols) >= 2:
            heatmap_cols = numeric_cols[:10]  
            heatmap_valid, _ = is_real_life_valid(
                "correlation heatmap", heatmap_cols, df, dataset_schema
            )
            if heatmap_valid:
                heatmap_dict = {
                    "graph": "correlation heatmap",
                    "columns": heatmap_cols,
                    "analysis_type": "multivariate",
                    "relationship_strength": 0.0,
                    "relationship_method": "multivariate",
                }
                is_redundant, _ = is_redundant_graph(heatmap_dict, valid_graphs, df)
                if not is_redundant:
                    heatmap_dict["score"] = score_graph_importance(heatmap_dict, df) + 2.0  # boost
                    valid_graphs.append(heatmap_dict)
                    logger.debug("  → Correlation heatmap added")

        if len(numeric_cols) >= 3 and corr_matrix is not None:
            correlations = []
            for c1, c2 in combinations(numeric_cols, 2):
                if c1 in corr_matrix.columns and c2 in corr_matrix.columns:
                    strength = corr_matrix.loc[c1, c2]
                    if pd.notna(strength) and abs(strength) >= config.MIN_CORRELATION:
                        correlations.append((c1, c2, abs(strength)))

            correlations.sort(key=lambda x: x[2], reverse=True)

            seen_triplets: Set[tuple] = set()
            for c1, c2, _ in correlations[:10]:
                for c3 in numeric_cols:
                    if c3 in {c1, c2}:
                        continue
                    triplet = tuple(sorted([c1, c2, c3]))
                    if triplet in seen_triplets:
                        continue
                    if c1 not in corr_matrix.columns or c3 not in corr_matrix.columns:
                        continue
                    strength1 = corr_matrix.loc[c1, c3]
                    strength2 = corr_matrix.loc[c2, c3]
                    has_rel1 = pd.notna(strength1) and strength1 >= config.MIN_CORRELATION
                    has_rel2 = pd.notna(strength2) and strength2 >= config.MIN_CORRELATION

                    if has_rel1 or has_rel2:
                        seen_triplets.add(triplet)
                        graph_info = get_graph_suggestions(list(triplet), metadata)
                        for graph in graph_info["suggested_graphs"]:
                            is_valid, _ = is_real_life_valid(graph, list(triplet), df, dataset_schema)
                            if not is_valid:
                                continue
                            graph_dict = {
                                "graph": graph,
                                "columns": list(triplet),
                                "analysis_type": "multivariate",
                                "relationship_strength": 0.0,
                                "relationship_method": "multivariate",
                            }
                            is_redundant, _ = is_redundant_graph(graph_dict, valid_graphs, df)
                            if not is_redundant:
                                graph_dict["score"] = score_graph_importance(graph_dict, df)
                                valid_graphs.append(graph_dict)

                if len(seen_triplets) >= 5:
                    break

        logger.debug(f"Total graphs before selection: {len(valid_graphs)}")


    # SMART SELECTION 
    valid_graphs.sort(key=lambda x: x["score"], reverse=True)

    selected_graphs = []
    seen_univariate_cols: Set[str] = set()
    seen_bivariate_pairs: Set[frozenset] = set()
    univariate_count = 0
    bivariate_count = 0
    multivariate_added = 0

    for graph in valid_graphs:
        if graph["graph"] == "correlation heatmap":
            selected_graphs.append(graph)
            break

    for graph in valid_graphs:
        if graph["analysis_type"] != "univariate":
            continue
        if univariate_count >= config.MAX_UNIVARIATE:
            break
        col_key = graph["columns"][0]
        if col_key in seen_univariate_cols:
            continue
        seen_univariate_cols.add(col_key)
        selected_graphs.append(graph)
        univariate_count += 1

    for graph in valid_graphs:
        if graph["analysis_type"] != "bivariate":
            continue
        if bivariate_count >= config.MAX_BIVARIATE:
            break
        pair_key = frozenset(graph["columns"])
        if pair_key in seen_bivariate_pairs:
            continue
        seen_bivariate_pairs.add(pair_key)
        selected_graphs.append(graph)
        bivariate_count += 1

    for graph in valid_graphs:
        if graph["analysis_type"] != "multivariate":
            continue
        if graph["graph"] == "correlation heatmap":
            continue
        if multivariate_added >= config.MAX_MULTIVARIATE:
            break
        selected_graphs.append(graph)
        multivariate_added += 1

    selected_graphs.sort(key=lambda x: x["score"], reverse=True)
    logger.debug(f"Returning {len(selected_graphs)} selected graphs")
    return selected_graphs



# UTILITY


def print_graph_summary(graphs: List[Dict], top_n: int = 20):
    logger.info(f"\n{'='*80}")
    logger.info(f"TOP {min(top_n, len(graphs))} RECOMMENDED GRAPHS")
    logger.info(f"{'='*80}\n")
    for i, graph in enumerate(graphs[:top_n], 1):
        logger.info(f"{i}. {graph['graph']}")
        logger.info(f"   Columns: {', '.join(graph['columns'])}")
        logger.info(f"   Type: {graph['analysis_type']}")
        logger.info(f"   Score: {graph['score']:.2f}")
        if graph["relationship_strength"] > 0:
            logger.info(
                f"   Relationship: {graph['relationship_method']} "
                f"(strength: {graph['relationship_strength']:.3f})"
            )
        logger.info()