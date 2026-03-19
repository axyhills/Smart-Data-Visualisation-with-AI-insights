# Smart Data Visualization with AI Insights

This project is an experimental **AI-assisted exploratory data analysis tool** designed to help users quickly understand datasets through automated visualizations and AI-generated insights.

Instead of manually creating dozens of charts during exploratory data analysis, the system analyzes the dataset structure, identifies statistically meaningful relationships between variables, and recommends the most relevant visualizations. It can also generate short natural-language insights explaining what the data shows.

The goal of this project is to simplify the early stages of data analysis and make dataset exploration more accessible.

---

## Key Features

### Automated Data Exploration
The system analyzes uploaded datasets and identifies possible relationships between variables.

### Intelligent Graph Recommendation
Charts are generated based on column types and statistical relationships.

### Statistical Validation
Before generating visualizations, the system performs statistical checks such as correlation analysis and variance tests to avoid misleading graphs.

### AI Generated Insights
For each visualization, the system can generate short explanations describing trends, correlations, or anomalies in the data.

### Interactive Dashboard
Users can explore graphs, build custom visualizations, and interact with the dataset through a simple web interface.

### Conversational Data Interaction
A chatbot interface allows users to ask questions about the dataset and receive AI-generated explanations.

---

## System Architecture
The application follows a modular design where each component handles a specific task in the analysis pipeline.

Dataset Upload
      ↓
Data Cleaning & Preprocessing
      ↓
Column Profiling
      ↓
Statistical Relationship Detection
      ↓
Graph Recommendation Engine
      ↓
Visualization Rendering
      ↓
AI Insight Generation


---

## Tech Stack

**Backend**
- Python
- Pandas
- NumPy
- SciPy

**Visualization**
- Plotly

**Interface**
- Streamlit

**AI / LLM Integration**
- Ollama 

---

## Project Structure
project/
│
├── main.py
├── graph_generator.py
├── ai_engine.py
├── graph_mapping.py
├── plot_graph.py
├── utils.py
├── config.py
│
└── data/

**main.py**  
Main application entry point.

**graph_generator.py**  
Responsible for detecting variable relationships and selecting appropriate graphs.

**ai_engine.py**  
Handles AI-based insight generation.

**graph_mapping.py**  
Defines mappings between dataset column types and visualizations.

**plot_graph.py**  
Generates the visualizations.

**utils.py**  
Utility functions used across the project.

---

## Example Workflow

1. Upload a dataset (CSV format)
2. The system analyzes column types and statistics
3. Relevant graphs are generated automatically
4. Users explore charts or create their own
5. AI generates insights explaining the visualizations

---

## Future Improvements

- Support for Excel and database datasets
- More statistical validation techniques
- Improved graph ranking algorithms
- Better AI insight generation
- Cloud deployment support

---

## Disclaimer

This project is an experimental prototype for exploring automated data analysis workflows. AI-generated insights should always be validated before making decisions.
