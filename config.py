from pydantic import BaseModel,ConfigDict, Field
from typing import List, Optional

# CORE SIGNAL THRESHOLDS


MIN_VARIANCE_THRESHOLD = 1e-4          # Filter near-constant numeric columns
MIN_VARIANCE_RATIO = 0.01              # 1% variance relative to range
MIN_CORRELATION_THRESHOLD = 0.3        # Absolute Pearson threshold
MIN_CORRELATION = 0.3                  # Absolute Pearson threshold
MIN_PVALUE = 0.05                      # Standard statistical significance
MIN_EFFECT_SIZE = 0.2                  # Small-to-medium effect
MIN_CRAMERS_V = 0.25                   # Moderate categorical association
MIN_MI_SCORE = 0.05                    # Mutual information lower bound



# DATA QUALITY FILTERS


MAX_MISSING_RATIO = 0.3                # Drop columns >30% missing
MAX_OUTLIER_RATIO = 0.2                # >20% extreme outliers = unstable
MAX_CARDINALITY = 25                   # Prevent noisy high-cardinality categories



# SAMPLE SIZE SAFETY LIMITS


MIN_HISTOGRAM_SAMPLES = 30             # Central Limit Theorem safety
MIN_SCATTER_SAMPLES = 20
MIN_HEATMAP_SAMPLES = 40



# GRAPH SELECTION LIMITS

MAX_SAMPLE_ROWS = 5000
MAX_GRAPHS = 15

MAX_UNIVARIATE = 5
MAX_BIVARIATE = 7
MAX_MULTIVARIATE = 3

MAX_COLUMNS_FOR_COMBO = 10             # Prevent combinatorial explosion
MAX_NUMERIC_FOR_TRIVARIATE = 6



# BUSINESS INSIGHT SETTINGS


KPI_TOP_N = 5
TREND_WINDOW = 3

class ColumnSchema(BaseModel):
    """
    Describes a single column in the dataset
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(..., description="Column name as in the CSV")

    detected_type: str = Field(
        ...,
        description="Detected semantic type: numerical, categorical, datetime, text, id, boolean"
    )

    missing_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of missing values in the column"
    )

    unique_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of unique values in the column"
    )


# DATASET SCHEMA
class DatasetSchema(BaseModel):
    """
    Describes the entire dataset
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    row_count: int = Field(..., ge=0, description="Number of rows in dataset")
    column_count: int = Field(..., ge=0, description="Number of columns in dataset")

    columns: List[ColumnSchema] = Field(
        ..., description="Metadata for each column"
    )

    cleaning_summary: List[str] = Field(
        default_factory=list,
        description="Human-readable summary of cleaning steps applied"
    )
