from unicodedata import numeric

import pandas as pd
import numpy as np
from typing import List, Tuple
import re

from config import ColumnSchema, DatasetSchema, MAX_SAMPLE_ROWS


# 1️⃣ RULE-BASED COLUMN TYPE DETECTION

def detect_column_type(series: pd.Series) -> str:
    col_name = series.name.lower().strip()
   
    if (
        re.search(r'(^id$)|(_id$)|(^id_)|(id$)', col_name)
        or "uuid" in col_name
    ):
        return "id"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return "datetime"
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d\.-]", "", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        has_mixed = series.astype(str).str.contains(
        r'\d+[a-zA-Z/\\-]+|[a-zA-Z]+\d+', regex=True, na=False).mean()

        if numeric.notna().mean() > 0.8 and has_mixed < 0.2:
            return "numerical"
        
        

        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if unique_ratio < 0.1:
            return "categorical"

        return "text"

    return "text"


# 2️⃣ COLUMN PROFILING 

def profile_columns(df: pd.DataFrame) -> List[ColumnSchema]:
    column_schemas = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]

        missing_percent = round(series.isna().mean() * 100, 2)
        unique_percent = round(
            series.nunique(dropna=True) / max(total_rows, 1) * 100, 2
        )

        detected_type = detect_column_type(series)

        column_schemas.append(
            ColumnSchema(
                name=col,
                detected_type=detected_type,
                missing_percent=missing_percent,
                unique_percent=unique_percent,

            )
        )

    return column_schemas



# 3️⃣ DEPENDENCY-BASED MISSING VALUE FILLING 


def dependency_fill(df: pd.DataFrame, target_col: str) -> pd.Series:
    if df[target_col].isna().sum() == 0:
        return df[target_col]

    candidate_groups = [
        c for c in df.columns
        if c != target_col
        and df[c].nunique(dropna=True) < 20
        and df[c].isna().mean() < 0.3
    ]

    best_series = df[target_col]

    for group_col in candidate_groups:
        grouped = df.groupby(group_col)[target_col]

        if pd.api.types.is_numeric_dtype(df[target_col]):
            filled = best_series.fillna(grouped.transform("median"))
        else:
            filled = best_series.fillna(
                grouped.transform(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
            )

        if filled.isna().sum() < best_series.isna().sum():
            best_series = filled

    return best_series


# 4️⃣ CLEAN DATAFRAME 

def clean_dataframe(
    df: pd.DataFrame,
    column_schemas: List[ColumnSchema]
) -> Tuple[pd.DataFrame, List[str]]:

    clean_df = df.copy()
    cleaning_summary = []

    # Dependency-based filling
    for col_schema in column_schemas:
        col = col_schema.name
        before = clean_df[col].isna().sum()

        clean_df[col] = dependency_fill(clean_df, col)

        after = clean_df[col].isna().sum()
        if after < before:
            cleaning_summary.append(
                f"Filled {before - after} missing values in '{col}' using dependency-based inference"
            )

    #  Statistical filling
    for col_schema in column_schemas:
        col = col_schema.name
        col_type = col_schema.detected_type

        if col_type == "numerical":
            cleaned = (
                clean_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.-]", "", regex=True)
            )
            clean_df[col] = pd.to_numeric(cleaned, errors="coerce")

            if clean_df[col].isna().any():
                median = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median)
                cleaning_summary.append(
                    f"Filled remaining missing values in '{col}' with median"
                )

        elif col_type == "categorical":
            mode = clean_df[col].mode(dropna=True)
            if not mode.empty:
                clean_df[col] = clean_df[col].fillna(mode[0])
                cleaning_summary.append(
                    f"Filled missing values in '{col}' with mode"
                )

    # Remove duplicates
    before = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    removed = before - len(clean_df)
    if removed > 0:
        cleaning_summary.append(f"Removed {removed} duplicate rows")

    #  Clean categorical text
    for col_schema in column_schemas:
        if col_schema.detected_type == "categorical":
            col = col_schema.name
            clean_df[col] = (
                clean_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    #  Final validation
    clean_df.replace(["nan", "none", "null", ""], np.nan, inplace=True)

    return clean_df, cleaning_summary


# 5️⃣ MASTER PIPELINE

def prepare_dataset(df: pd.DataFrame) -> dict:
    raw_df = df  

    

    initial_schemas = profile_columns(df)
    clean_df, cleaning_summary = clean_dataframe(raw_df, initial_schemas)
    column_schemas = profile_columns(clean_df)
    if len(clean_df) > MAX_SAMPLE_ROWS:
        analysis_df = clean_df.sample(
            n=MAX_SAMPLE_ROWS,
            random_state=42
        ).reset_index(drop=True)
    else:
        analysis_df = clean_df.copy()
    

    dataset_schema = DatasetSchema(
        row_count=len(clean_df),
        column_count=clean_df.shape[1],
        columns=column_schemas,
        
        cleaning_summary=cleaning_summary
        
    )
    numeric_cols = None

    numeric_cols = [
    col.name for col in dataset_schema.columns
    if col.detected_type == "numerical" 
    ]

    corr_matrix = None
    if len(numeric_cols) > 1:
        corr_matrix = analysis_df[numeric_cols].corr(method="pearson").abs()

    precomputed_stats = {
        "corr_matrix": corr_matrix,
        "numeric_cols": numeric_cols
}
    return {
        "raw_df": df,
        "cleaned_df": clean_df,
        "analysis_df": analysis_df,
        "dataset_schema": dataset_schema,
        "precomputed_stats": precomputed_stats
    }
