from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import concurrent.futures

# LLM 
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model="llama3.2:1b", temperature=0.3)
    return _llm

# Prompt 
INSIGHT_PROMPT = PromptTemplate(
    input_variables=["graph_type", "columns", "stats"],
    template="""You are a senior data analyst.

Given:
- Graph type: {graph_type}
- Columns involved: {columns}
- Summary statistics:
{stats}

Task:
1. Identify the key pattern or trend, citing specific numbers from the statistics above.
2. Highlight the most extreme values (min, max, outliers) and quantify any spread or skew.
3. If comparing columns, state the numerical difference or ratio.
4. Explain what these numbers could mean in a real-world or business context.

Rules:
- Be concise (4-6 sentences)
- Every claim MUST reference a specific number from the statistics provided
- Do NOT invent or estimate values not present in the statistics
- If the data is insufficient for a numerical insight, say so clearly
""",
)
def _call_llm(prompt: str) -> str:
    return _get_llm().invoke(prompt).strip()

# Public API 
TIMEOUT_SECONDS = 60 
import pandas as pd

def build_stats_string(df: pd.DataFrame, columns: list, corr_matrix=None) -> str:
    """
    Generate a rich numerical summary for the given columns.
    Pass the result as the `stats` argument to generate_ai_insight.
    Accepts an optional precomputed corr_matrix to avoid recalculation.
    """
    lines = []
    subset = df[columns].select_dtypes(include="number")

    for col in subset.columns:
        s = subset[col].dropna()
        if s.empty:
            continue
        q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
        lines.append(
            f"{col}:\n"
            f"  count={len(s)}, mean={s.mean():.4g}, std={s.std():.4g}\n"
            f"  min={s.min():.4g}, max={s.max():.4g}\n"
            f"  Q1={q1:.4g}, median={median:.4g}, Q3={q3:.4g}\n"
            f"  skewness={s.skew():.4g}, null_count={df[col].isna().sum()}"
        )

    numeric_in_selection = subset.columns.tolist()
    if len(numeric_in_selection) > 1:
        if corr_matrix is not None:
            available = [c for c in numeric_in_selection if c in corr_matrix.index]
            if len(available) > 1:
                sliced = corr_matrix.loc[available, available].round(3)
                lines.append(f"\nCorrelation matrix:\n{sliced.to_string()}")
        else:
            corr = subset.corr().round(3)
            lines.append(f"\nCorrelation matrix:\n{corr.to_string()}")
    cat_cols = df[columns].select_dtypes(exclude="number").columns
    for col in cat_cols:
        vc = df[col].value_counts().head(5)
        lines.append(
            f"{col} (categorical):\n"
            f"  unique={df[col].nunique()}, top_values={vc.to_dict()}"
    )

    return "\n".join(lines) if lines else "No numeric columns available."

def generate_ai_insight(
    graph_type: str,
    columns: list,
    stats: str,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """
    Call Ollama llama3.2:1b in a background thread with a hard timeout.
    Never blocks the Streamlit main thread indefinitely.
    """
    prompt = INSIGHT_PROMPT.format(
        graph_type=graph_type,
        columns=", ".join(columns),
        stats=stats,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_llm, prompt)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return (
                f"\u23f1\ufe0f **Insight timed out** after {timeout}s.\n\n"
                "Ollama is still loading the model or your machine is under load. "
                "Try again in a few seconds, or run `ollama pull llama3.2:1b` to ensure "
                "the model is fully downloaded."
            )
        except Exception as e:
            return (
                f"\u26a0\ufe0f **AI insight unavailable.**\n\n"
                f"**Reason:** {e}\n\n"
                "**Fix:** Make sure Ollama is running (`ollama serve`) "
                "and llama3.2:1b is pulled (`ollama pull llama3.2:1b`)."
            )