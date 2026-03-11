from langchain_ollama import OllamaLLM
import pandas as pd
import json

# LLM instance 
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model="llama3.2:1b", temperature=0.4)
    return _llm


# Dataset context builder
def build_dataset_context(
    analysis_df: pd.DataFrame,
    dataset_schema,
    ranked_graphs: list = None,
    current_chart: dict = None,
) -> str:
    """
    Builds a compact text context about the dataset to inject into every prompt.
    Keeps token count low by summarising rather than dumping raw data.
    """
    lines = []

    # Basic shape
    lines.append(f"DATASET: {dataset_schema.row_count} rows, {dataset_schema.column_count} columns.")

    # Column summary
    col_summaries = []
    for col in dataset_schema.columns:
        col_summaries.append(f"  - {col.name} ({col.detected_type})")
    lines.append("COLUMNS:\n" + "\n".join(col_summaries))

    # Numeric stats 
    num_cols = [
        c.name for c in dataset_schema.columns
        if c.detected_type == "numerical"
    ]
    if num_cols:
        try:
            stats = analysis_df[num_cols].describe().round(2)
            lines.append("NUMERIC STATS (sample):\n" + stats.to_string())
        except Exception:
            pass

    # Categorical value counts (top 5 per col, max 3 cols)
    cat_cols = [
        c.name for c in dataset_schema.columns
        if c.detected_type == "categorical"
    ][:3]
    for col in cat_cols:
        try:
            top = analysis_df[col].value_counts().head(5).to_dict()
            lines.append(f"TOP VALUES in '{col}': {top}")
        except Exception:
            pass

    # Ranked graphs available
    if ranked_graphs:
        graph_list = [
            f"  - {g['graph']} on [{', '.join(g['columns'])}]"
            for g in ranked_graphs[:8]
        ]
        lines.append("RECOMMENDED GRAPHS:\n" + "\n".join(graph_list))

    # Currently viewed chart
    if current_chart:
        lines.append(
            f"USER IS CURRENTLY VIEWING: "
            f"{current_chart.get('chart_id', 'unknown')} "
            f"on columns [{', '.join(current_chart.get('columns', []))}]"
        )

    return "\n\n".join(lines)


#  System prompt
SYSTEM_PROMPT = """You are a smart BI assistant embedded in a data visualization app.

You have access to the following context about the user's dataset:

{context}

Your capabilities:
1. Answer questions about the dataset (columns, distributions, relationships).
2. Suggest which graphs to look at and why.
3. Explain what a chart the user is currently viewing means.
4. Answer general data analysis questions.

Rules:
- Be concise and direct (2-4 sentences unless more detail is needed).
- Base answers on the dataset context above — do not hallucinate column names or values.
- If you don't know something from the context, say so clearly.
- When suggesting graphs, refer to the RECOMMENDED GRAPHS list if available.
- If asked about the currently viewed chart, focus your explanation on it.
"""

def build_full_prompt(
    chat_history: list,
    user_message: str,
    context: str,
) -> str:
    """
    Builds the full prompt string from system prompt + history + new message.
    llama3.2:1b doesn't use chat templates via langchain, so we format manually.
    """
    system = SYSTEM_PROMPT.format(context=context)

    history_text = ""
    for msg in chat_history[-6:]:   # last 6 messages = 3 turns of context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    return (
        f"{system}\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"User: {user_message}\n"
        f"Assistant:"
    )


#Public API
def chat(
    user_message: str,
    chat_history: list,
    analysis_df: pd.DataFrame,
    dataset_schema,
    ranked_graphs: list = None,
    current_chart: dict = None,
    timeout: int = 45,
) -> str:
    """
    Send a message to the chatbot and get a response.

    Parameters
    ----------
    user_message   : the user's latest message
    chat_history   : list of {"role": "user"/"assistant", "content": str}
    analysis_df    : cleaned dataset
    dataset_schema : DatasetSchema object
    ranked_graphs  : output of generate_ranked_insights (optional)
    current_chart  : {"chart_id": str, "columns": list} of chart being viewed (optional)
    timeout        : seconds before giving up

    Returns
    -------
    str : assistant response
    """
    import concurrent.futures

    try:
        context = build_dataset_context(
            analysis_df, dataset_schema, ranked_graphs, current_chart
        )
        prompt = build_full_prompt(chat_history, user_message, context)

        def _invoke():
            return _get_llm().invoke(prompt).strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return (
                    f"⏱️ Response timed out after {timeout}s. "
                    "llama3.2:1b is still loading — try again in a moment."
                )

    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            return (
                "⚠️ Cannot reach Ollama. Make sure it's running:\n"
                "`ollama serve` in a terminal, then `ollama pull llama3.2:1b`"
            )
        return f"⚠️ Error: {err}"