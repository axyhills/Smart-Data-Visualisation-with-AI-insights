import streamlit as st
import pandas as pd
from datetime import datetime
from graph_mapping import get_graph_suggestions
from utils import prepare_dataset
from plot_graph import plot_graph
from ai_engine import generate_ai_insight, build_stats_string  as generate_stats
from graph_generator import generate_ranked_insights, is_real_life_valid
from chatbot import chat as chatbot_chat


# PAGE CONFIG

st.set_page_config(page_title="Smart Visualization", layout="wide")


# SESSION STATE INIT

defaults = {
    "raw_df": None,
    "prepared": None,
    # Explore page
    "explore_fig": None,
    "explore_insight": None,
    "explore_columns": [],
    "explore_chart_id": None,
    "alternative_graphs": {},
    "active_graph_selection": {},
    "active_analysis_section": None,
    "ranked_graphs": None,
    # Manual builder page
    "manual_fig": None,
    "manual_insight": None,
    "manual_columns": [],
    "manual_chart_id": None,
    "manual_selected_columns": [],
    # Dashboard
    "dashboard_items": [],
    # Chatbot
    "chat_history": [],
    "current_chart": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# CHART DEFINITIONS  

chart_definitions ={
    "histogram":            {"label": "📊 Histogram",            "requires": {"numerical": 1}},
    "box plot":             {"label": "📦 Box Plot",              "requires": {"numerical": 1}},
    "violin plot":          {"label": "🎻 Violin Plot",           "requires": {"numerical": 1}},
    "density plot":         {"label": "📈 Density Plot",          "requires": {"numerical": 1}},
    "bar chart":            {"label": "📊 Bar Chart",             "requires": {"categorical": 1}},
    "pie chart":            {"label": "🥧 Pie Chart",             "requires": {"categorical": 1}},
    "scatter plot":         {"label": "🎯 Scatter Plot",          "requires": {"numerical": 2}},
    "line plot":            {"label": "📈 Line Plot",             "requires": {"datetime": 1, "numerical": 1}},
    "area plot":            {"label": "📈 Area Plot",             "requires": {"datetime": 1, "numerical": 1}},
    "hexbin plot":          {"label": "⬡ Hexbin Plot",           "requires": {"numerical": 2}},
    "grouped box plot":     {"label": "📦 Grouped Box Plot",      "requires": {"categorical": 1, "numerical": 1}},
    "grouped violin plot":  {"label": "🎻 Grouped Violin Plot",   "requires": {"categorical": 1, "numerical": 1}},
    "aggregated bar chart": {"label": "📊 Aggregated Bar Chart",  "requires": {"categorical": 1, "numerical": 1}},
    "correlation heatmap":  {"label": "🌡️ Correlation Heatmap",  "requires": {"numerical": "2+"}},
    "pair plot":            {"label": "🔗 Pair Plot",             "requires": {"numerical": "2+"}},
    }


# SIDEBAR

with st.sidebar:
    st.header("🧭 Navigation")
    pages = ["Home", "Explore Graphs", "Build Your Own Chart", "Dashboard", "💬 AI Chat"]
    default_idx = pages.index(st.session_state.get("active_page", "Home"))
    prev_page = st.session_state.get("active_page", "Home")
    page = st.radio("Select Page", pages, index=default_idx)
    if page != prev_page:
        st.session_state.active_page = page  
    elif st.session_state.get("force_reload", False):
        st.session_state.force_reload = False
        st.rerun()
    st.divider()
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        
        if (
            st.session_state.raw_df is None
            or not df_raw.equals(st.session_state.raw_df)
        ):
            st.session_state.raw_df = df_raw
           
            st.session_state.prepared = None
            st.session_state.alternative_graphs = {}
            st.session_state.active_graph_selection = {}
            st.session_state.ranked_graphs_by_section = {}  

        if st.session_state.prepared is None:
            with st.spinner("Preparing dataset…"):
                st.session_state.prepared = prepare_dataset(df_raw)
        st.success("Dataset loaded ✅")


# SAFETY CHECK

if st.session_state.raw_df is None:
    st.info("👈 Upload a CSV from the sidebar to begin.")
    st.stop()

raw_df          = st.session_state.raw_df
prepared        = st.session_state.prepared
analysis_df     = prepared["analysis_df"]
dataset_schema  = prepared["dataset_schema"]
precomputed     = prepared.get("precomputed_stats", {})
corr_matrix     = precomputed.get("corr_matrix")
numeric_cols    = precomputed.get("numeric_cols", [])


# HELPERS

def detect_column_types(schema):
    num, cat, dt = [], [], []
    for col in schema.columns:
        
        if col.detected_type == "numerical":   num.append(col.name)
        elif col.detected_type == "categorical": cat.append(col.name)
        elif col.detected_type == "datetime":    dt.append(col.name)
    return num, cat, dt

numerical_cols, categorical_cols, datetime_cols = detect_column_types(dataset_schema)

def check_availability(requires, numerical_cols, categorical_cols, datetime_cols):
    for col_type, count in requires.items():
        if col_type == "numerical":
            needed = int(str(count).replace("+", ""))
            if len(numerical_cols) < needed:
                return False
        elif col_type == "categorical":
            if len(categorical_cols) < count:
                return False
        elif col_type == "datetime":
            if len(datetime_cols) < count:
                return False
    return True


def get_alternative_graphs(columns):
    metadata = [
        {"name": col.name, "detected_type": col.detected_type}
        for col in dataset_schema.columns
        
    ]
    graph_info = get_graph_suggestions(columns, metadata)
    return [
        g for g in graph_info["suggested_graphs"]
        if is_real_life_valid(g, columns, analysis_df, dataset_schema)[0]
    ]


def render_chart(fig):
    st.plotly_chart(fig, use_container_width=True)





# HOME

def show_home():
    st.title("📊 Smart BI")
    st.subheader("📌 Dataset Overview")

    c1, c2 = st.columns(2)
    c1.metric("Rows", dataset_schema.row_count)
    c2.metric("Columns", dataset_schema.column_count)

    with st.expander("🧬 Column Profile & Type Detection", expanded=True):
        combined_df = pd.DataFrame([
            {
                "Column": col.name,
                "Pandas dtype": raw_df[col.name].dtype,
                "Detected type": col.detected_type,
                "Missing": raw_df[col.name].isna().sum(),
                "Missing %": round(raw_df[col.name].isna().mean() * 100, 2),
            }
            for col in dataset_schema.columns
           
        ])
        st.dataframe(combined_df, use_container_width=True)

    st.subheader("🔍 Preview")
    n = st.slider("Rows to preview", 5, min(100, len(raw_df)), 10, 5)
    st.dataframe(raw_df.head(n), use_container_width=True)

    with st.expander("🧹 Cleaning Summary"):
        for line in dataset_schema.cleaning_summary:
            st.caption("• " + line)

    st.subheader("⬇️ Download Cleaned Dataset")
    st.download_button(
        "📥 Download cleaned data as CSV",
        data=analysis_df.to_csv(index=False).encode(),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )


# EXPLORE 
_SECTION_TYPES = {
    "distribution": ["univariate"],
    "relation":     ["bivariate", "time_series"],
    "multivariate": ["multivariate"],
}

def get_ranked_graphs(active_section: str = None):
    """
    Lazy, section-aware graph generation.
    Only computes graphs needed for the active section on first click.
    Results are cached per section in session_state so re-clicks are instant.
    """
    if "ranked_graphs_by_section" not in st.session_state:
        st.session_state.ranked_graphs_by_section = {}

    if active_section is None:
        all_graphs = []
        for graphs in st.session_state.ranked_graphs_by_section.values():
            all_graphs.extend(graphs)
        return all_graphs

    if active_section in st.session_state.ranked_graphs_by_section:
        return st.session_state.ranked_graphs_by_section[active_section]

    needed_types = _SECTION_TYPES.get(active_section, ["univariate"])

    with st.spinner(f"Finding best graphs for this section…"):
        result = generate_ranked_insights(
            analysis_df, dataset_schema, precomputed,
            only_types=needed_types
        )
        st.session_state.ranked_graphs_by_section[active_section] = result

    if not result:
        st.warning(
            "No graphs found for this section. Check Home → Column Profile "
            "to verify column types were detected correctly."
        )

    return result


def explore_section():
    st.title("🔍 Explore Graphs")
    st.subheader("What would you like to explore?")

    section_labels = {
        "distribution": "📊 Understand Your Data",
        "relation":     "🔍 Find Connections",
        "multivariate": "🌐 Explore Bigger Patterns",
    }

    btn_cols = st.columns(3)
    for i, (key, label) in enumerate(section_labels.items()):
        with btn_cols[i]:
            is_active = st.session_state.active_analysis_section == key
            btn_style = "primary" if is_active else "secondary"
            icon = "▼" if is_active else "▶"
            if st.button(
                f"{icon} {label}",
                key=f"section_{key}",
                use_container_width=True,
                type=btn_style,
            ):
                if is_active:
                    st.session_state.active_analysis_section = None
                else:
                    st.session_state.active_analysis_section = key
                st.rerun()

    active = st.session_state.active_analysis_section
    if not active:
        return

    all_graphs = get_ranked_graphs(active_section=active)

    section_mapping = {
        "univariate":   "distribution",
        "bivariate":    "relation",
        "multivariate": "multivariate",
        "time_series":  "relation",
    }

    seen_combos: set = set()
    section_graphs = []
    for g in all_graphs:
        mapped = section_mapping.get(g.get("analysis_type", ""), "multivariate")
        if mapped != active:
            continue
        dedup_key = (frozenset(g["columns"]), g["graph"])
        if dedup_key in seen_combos:
            continue
        seen_combos.add(dedup_key)
        section_graphs.append(g)

    if not section_graphs:
        st.info(f"No graphs available for **{section_labels[active]}** with the current dataset.")
        st.caption("Try checking Home → Column Profile to verify column types are detected correctly.")
        return

    st.divider()
    st.subheader(section_labels[active])
    st.caption(f"{len(section_graphs)} graph(s) found")

    for pair_start in range(0, len(section_graphs), 2):
        pair = section_graphs[pair_start : pair_start + 2]
        if len(pair) == 1 or pair[0]["graph"] in ("correlation heatmap", "pair plot"):
            chart_cols = st.columns(1)
        else:
            chart_cols = st.columns(2)

        for col_idx, g in enumerate(pair):
            if len(chart_cols) == 1 and col_idx > 0:
                break

            idx = pair_start + col_idx
            graph_key = f"{active}_{idx}"

            # Cache alternatives 
            if graph_key not in st.session_state.alternative_graphs:
                if g["graph"] == "correlation heatmap" and len(g["columns"]) > 2:
                    st.session_state.alternative_graphs[graph_key] = ["correlation heatmap"]
                else:
                    st.session_state.alternative_graphs[graph_key] = get_alternative_graphs(g["columns"])

            alternatives = st.session_state.alternative_graphs[graph_key]
            if not alternatives:
                alternatives = [g["graph"]]

            # Default selection
            if graph_key not in st.session_state.active_graph_selection:
                st.session_state.active_graph_selection[graph_key] = (
                    g["graph"] if g["graph"] in alternatives else alternatives[0]
                )

            selected_gt = st.session_state.active_graph_selection[graph_key]

            with chart_cols[min(col_idx, len(chart_cols)-1)]:
                try:
                    fig = plot_graph(analysis_df, selected_gt, g["columns"])
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.current_chart = {
                        "chart_id": selected_gt,
                        "columns": g["columns"],
                    }
                except Exception as e:
                    st.error(f"Could not render '{selected_gt}': {e}")

                if len(alternatives) > 1:
                    chosen = st.selectbox(
                        "Switch chart type",
                        alternatives,
                        index=alternatives.index(selected_gt) if selected_gt in alternatives else 0,
                        key=f"selector_{graph_key}",
                    )
                    st.session_state.active_graph_selection[graph_key] = chosen
                else:
                    st.caption(f"📊 {selected_gt}")

                
                
                
                if st.button("✨ AI Insight", key=f"ai_{graph_key}" ):
                    with st.spinner("Generating insight…"):
                            
                        col_data = analysis_df[g["columns"]].dropna()
                        stats_str = generate_stats(
                            col_data, g["columns"],corr_matrix=corr_matrix
                                
                            )
                        insight = generate_ai_insight(selected_gt, g["columns"], stats_str)
                        st.session_state[f"insight_{graph_key}"] = insight
                        st.rerun()
                    if st.session_state.get(f"insight_{graph_key}"):
                        st.write(st.session_state[f"insight_{graph_key}"])
                
               


                if st.button("💬 Ask AI ", key=f"ask_ai_{graph_key}"):
                    st.session_state.current_chart = {
                            "chart_id": selected_gt,
                            "columns": g["columns"],
                        }
                    st.session_state.active_page = "💬 AI Chat"
                    st.rerun()
                    if st.session_state.get(f"insight_{graph_key}"):
                        st.info(st.session_state[f"insight_{graph_key}"])
                
                if st.button("💾 Save", key=f"save_{graph_key}"):
                    try:
                        save_fig = plot_graph(analysis_df, selected_gt, g["columns"])
                        st.session_state.dashboard_items.append({
                                "figure":  save_fig,
                                "name":    selected_gt,
                                "insight": st.session_state.get(f"insight_{graph_key}", ""),
                                "columns": g["columns"],
                                "time":    datetime.now().strftime("%H:%M:%S"),
                            })
                        st.success("Saved!")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

                

                # with st.expander("💬 Ask about this chart", expanded=False):
                #     chat_key = f"chart_chat_{graph_key}"
                #     if chat_key not in st.session_state:
                #         st.session_state[chat_key] = []

                #     # Display existing messages
                #     for msg in st.session_state[chat_key]:
                #         with st.chat_message(msg["role"]):
                #             st.write(msg["content"])

                #     # Input
                #     user_q = st.chat_input(
                #         "Ask something about this chart…",
                #         key=f"chat_input_{graph_key}"
                #     )
                #     if user_q:
                #         st.session_state[chat_key].append(
                #             {"role": "user", "content": user_q}
                #         )
                #         with st.chat_message("user"):
                #             st.write(user_q)

                #         with st.chat_message("assistant"):
                #             with st.spinner("Thinking…"):
                #                 response = chatbot_chat(
                #                     user_message=user_q,
                #                     chat_history=st.session_state[chat_key][:-1],
                #                     analysis_df=analysis_df,
                #                     dataset_schema=dataset_schema,
                #                     ranked_graphs=None,
                #                     current_chart={
                #                         "chart_id": selected_gt,
                #                         "columns": g["columns"],
                #                     },
                #                 )
                #             st.write(response)

                #         st.session_state[chat_key].append(
                #             {"role": "assistant", "content": response}
                #         )

                #     if st.session_state[chat_key]:
                #         if st.button("🧹 Clear", key=f"clear_chart_chat_{graph_key}"):
                #             st.session_state[chat_key] = []
                #             st.rerun()
        st.divider()


# MANUAL BUILDER                        


def manual_builder():
    st.title("🎨 Build Your Own Chart")
    numerical_cols, categorical_cols, datetime_cols = detect_column_types(dataset_schema)

    grid = st.columns(3)
    for i, (chart_id, info) in enumerate(chart_definitions.items()):
        with grid[i % 3]:
            available = check_availability(
                info["requires"], numerical_cols, categorical_cols, datetime_cols
            )
            if available:
                if st.button(info["label"], key=f"manual_{chart_id}", use_container_width=True):
                    st.session_state.manual_chart_id = chart_id
                    st.session_state.manual_selected_columns = []
                    st.session_state.manual_fig = None
                    st.session_state.manual_insight = None
            else:
                st.button(info["label"], disabled=True, use_container_width=True)

    chart_id = st.session_state.manual_chart_id
    if not chart_id:
        return

    st.divider()
    requires = chart_definitions[chart_id]["requires"]

    # Build selectable pool from requires
    pool = []
    if "numerical" in requires:   pool += numerical_cols
    if "categorical" in requires: pool += categorical_cols
    if "datetime" in requires:    pool += datetime_cols
    pool = sorted(set(pool))

    selected_cols = st.multiselect(
        f"Select columns for **{chart_definitions[chart_id]['label']}**",
        pool,
        default=st.session_state.manual_selected_columns,
        key="manual_col_select",
    )
    st.session_state.manual_selected_columns = selected_cols

    if st.button("🚀 Generate Chart", key="generate_manual_chart"):
        # Validate counts
        n_num = sum(1 for c in selected_cols if c in numerical_cols)
        n_cat = sum(1 for c in selected_cols if c in categorical_cols)
        n_dt  = sum(1 for c in selected_cols if c in datetime_cols)

        ok = True
        for col_type, count in requires.items():
            needed = int(str(count).replace("+", ""))
            actual = {"numerical": n_num, "categorical": n_cat, "datetime": n_dt}.get(col_type, 0)
            if actual < needed:
                st.warning(f"Please select at least {needed} {col_type} column(s).")
                ok = False

        if ok:
            try:
                fig = plot_graph(analysis_df, chart_id, selected_cols)
                st.session_state.manual_fig     = fig
                st.session_state.manual_columns = selected_cols
                st.session_state.manual_chart_id_display = chart_id
                st.session_state.manual_insight = None  

                st.session_state.current_chart = {"chart_id": chart_id, "columns": selected_cols}
            except Exception as e:
                st.error(f"Could not render chart: {e}")
            
    if st.session_state.manual_fig:
        _output_section(
            fig=st.session_state.manual_fig,
            chart_name=st.session_state.manual_chart_id,
            columns=st.session_state.manual_columns,
            insight_key="manual_insight",
            save_key="manual_save",
        )


# OUTPUT SECTION  

def _output_section(fig, chart_name, columns, insight_key, save_key, corr_matrix=None):
    st.divider()
    st.subheader(f"📈 {chart_name}")
    render_chart(fig)

    dl_col, tip_col = st.columns(2)
    with dl_col:
        st.download_button(
            "⬇ Download as HTML",
            data=fig.to_html(),
            file_name=f"{chart_name}.html",
            mime="text/html",
            key=f"dl_{save_key}",
        )
    with tip_col:
        st.info("💡 Use the camera icon on the chart to save as PNG")

    # AI Insight
    st.subheader("🧠 AI Insight")
    if st.button("✨ Generate AI Insight", key=f"ai_{insight_key}"):
        with st.spinner("Generating insight…"):
            col_data = analysis_df[columns].dropna()
            stats = generate_stats(col_data, columns,corr_matrix=corr_matrix)
            st.session_state[insight_key] = generate_ai_insight(chart_name, columns, stats)

            

        if st.session_state.get(insight_key):
            st.write(st.session_state[insight_key])
    if st.button("💬 Ask AI ", key=f"ask_ai_{chart_name}"):  
            st.session_state.current_chart = {
                "chart_id": chart_name,   
                "columns": columns,   
            }
            st.session_state.active_page = "💬 AI Chat"
            st.rerun()
    # Save to Dashboard
    if st.button("💾 Save to Dashboard", key=f"save_{save_key}"):
        st.session_state.dashboard_items.append({
            "figure":  fig,
            "name":    chart_name,
            "insight": st.session_state.get(insight_key, ""),
            "columns": columns,
            "time":    datetime.now().strftime("%H:%M:%S"),
        })
        st.success("✅ Saved to Dashboard!")


# DASHBOARD

def show_dashboard():
    st.title("📊 Dashboard")

    if not st.session_state.dashboard_items:
        st.info("No charts saved yet. Generate charts and press 'Save to Dashboard'.")
        return

    for idx, item in enumerate(st.session_state.dashboard_items):
        st.markdown(f"### {item['name']}  `{item['time']}`")
        render_chart(item["figure"])

        if item.get("insight"):
            with st.expander("🧠 AI Insight"):
                st.write(item["insight"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Remove", key=f"remove_{idx}"):
                st.session_state.dashboard_items.pop(idx)
                st.rerun()
        with c2:
            st.download_button(
                "⬇ Download as HTML",
                data=item["figure"].to_html(),
                file_name=f"{item['name']}.html",
                mime="text/html",
                key=f"dl_dash_{idx}",
            )
        st.divider()

    if st.button("🧹 Clear All"):
        st.session_state.dashboard_items = []
        st.rerun()



# CHATBOT PAGE  

def show_chatbot():
    st.title("💬 AI Data Assistant")
    st.caption("Ask me anything about your dataset, charts, or data analysis.")

    current_chart = st.session_state.get("current_chart")
    if current_chart:
        st.info(
            f"📊 Context: currently viewing **{current_chart.get('chart_id')}** "
            f"on `{', '.join(current_chart.get('columns', []))}`"
        )

    if not st.session_state.chat_history:
        st.subheader("💡 Try asking:")
        suggestions = [
            "Explain the chart I am currently viewing.",
            "What are the most important columns in this dataset?",
            "Which graphs should I look at first?",
            "Are there any interesting patterns in the data?",
            "What does the distribution of numerical columns tell us?",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": suggestion}
                    )
                    with st.spinner("Thinking…"):
                        response = chatbot_chat(
                            user_message=suggestion,
                            chat_history=st.session_state.chat_history[:-1],
                            analysis_df=analysis_df,
                            dataset_schema=dataset_schema,
                            ranked_graphs=get_ranked_graphs(active_section=None),
                            current_chart=current_chart,
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your data…")
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = chatbot_chat(
                    user_message=user_input,
                    chat_history=st.session_state.chat_history[:-1],
                    analysis_df=analysis_df,
                    dataset_schema=dataset_schema,
                    ranked_graphs=get_ranked_graphs(active_section=None),
                    current_chart=current_chart,
                )
            st.write(response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

    
    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ROUTER

if page == "Home":
    show_home()
elif page == "Explore Graphs":
    explore_section()
elif page == "Build Your Own Chart":
    manual_builder()
elif page == "Dashboard":
    show_dashboard()
elif page == "💬 AI Chat":
    show_chatbot()