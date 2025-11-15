"""Prompts for the Financial Visualization Tool."""

# Stage 1: Data Summarization
SUMMARIZATION_PROMPT = """
You are a financial data analyst tasked with summarizing a dataset about Brazilian investment funds.

### Dataset Schema
{schema}

### Context
{context_str}

### Your Task
Analyze the dataset and provide a comprehensive summary that includes:

1. **Dataset Overview**:
   - Number of rows: {nrows}
   - Key columns and their purposes
   - Data types and ranges

2. **Financial Metrics Available**:
   - Performance indicators (returns, volatility, Sharpe ratio, etc.)
   - Fund characteristics (AUM, fees, risk ratings)
   - Temporal information (dates, periods)

3. **Data Quality Notes**:
   - Missing values and their significance
   - Outliers or unusual patterns
   - Distribution characteristics (skewness, spread)

4. **Insights for Visualization**:
   - Key relationships worth exploring
   - Interesting comparisons to visualize
   - Temporal trends available

{format_instructions}

Provide a detailed but concise summary focused on enabling effective financial visualizations.
"""

# Stage 2: Visualization Proposal
PROPOSAL_PROMPT = """
You are a financial visualization expert. Based on the dataset summary and user query, propose {n_visualizations} insightful visualization(s).

### Dataset Summary
{data_summary}

### User Context
{context_str}

### Guidelines for Financial Visualizations

1. **Chart Type Selection**:
   - Time series → Line charts
   - Comparisons → Bar charts or grouped bars
   - Distributions → Histograms or box plots
   - Correlations → Scatter plots
   - Compositions → Pie charts or stacked bars
   - Rankings → Horizontal bar charts

2. **Financial Best Practices**:
   - Show returns over time periods
   - Compare fund performance vs benchmarks
   - Display risk-return trade-offs
   - Highlight fee impacts
   - Show portfolio composition
   - Include reference lines for key thresholds

3. **Brazilian Fund Context**:
   - Consider ANBIMA classifications
   - Show performance in BRL context
   - Include CDI/SELIC comparisons when relevant
   - Display regulatory metrics (taxes, fees)

4. **Data Storytelling**:
   - Each visualization should answer a specific question
   - Focus on actionable insights
   - Make comparisons meaningful
   - Highlight outliers or notable patterns

{FORMAT_INSTRUCTIONS}

Propose {n_visualizations} visualization(s) that would provide the most value to a financial analyst or investor.
"""

# Stage 3: Code Generation
GENERATION_PROMPT = """
Your task is to generate Python code to create a financial visualization using {library}.

### Visualization Goal
Generate a {visualization}.
It must address: "{question}"
Use a graph title: "{short_question}"

### Data Summary
{data_summary}

### Context
{context_str}

### Code Template
You must modify this template:
```python
{library_template}
```

### Financial Visualization Requirements

1. **Code Structure**:
   - Only replace `<imports>` and `<stub>` placeholders
   - The `data` variable is already loaded as a pandas DataFrame
   - Do not write data loading code
   - Final line must assign the chart to the `chart` variable

2. **Financial Data Handling**:
   - Convert date columns properly: `pd.to_datetime(data['date'], errors='coerce')`
   - Handle percentage values (multiply by 100 if needed)
   - Format currency values with appropriate notation
   - Handle missing data gracefully

3. **Chart Aesthetics**:
   - Use professional financial styling
   - Clear axis labels with units (%, R$, etc.)
   - Legible font sizes
   - Color-blind friendly palettes
   - Include legends when using multiple series
   - Add reference lines for benchmarks (e.g., CDI rate)

4. **Brazilian Market Context**:
   - Format BRL currency: R$ X,XXX.XX
   - Show returns as percentages with 2 decimal places
   - Use Portuguese labels when appropriate
   - Consider ANBIMA fund classifications

5. **Data Quality**:
   - Remove NaN/null values before plotting
   - Handle outliers appropriately
   - Note any data limitations in the title/subtitle

### Response Format
1. First, provide a solution plan:
   - Data transformations needed
   - Fields to use
   - Visualization approach
   - Key aesthetics

2. Then provide the complete code:
```python
<your filled-in code here>
```

### Library-Specific Instructions
{library_instructions}

Generate production-quality code for financial analysis.
"""

# Library-specific instructions
MATPLOTLIB_INSTRUCTIONS = """
**Matplotlib/Seaborn Instructions**:
- Import: `import matplotlib.pyplot as plt` and `import seaborn as sns`
- Create figure: `fig, ax = plt.subplots(figsize=(12, 6))`
- Use seaborn styles: `sns.set_style('whitegrid')`
- Format axes: `ax.set_xlabel()`, `ax.set_ylabel()`, `ax.set_title()`
- Add grid: `ax.grid(True, alpha=0.3)`
- Rotate labels if needed: `plt.xticks(rotation=45, ha='right')`
- Final variable must be: `chart = fig`
"""

PLOTLY_INSTRUCTIONS = """
**Plotly Instructions**:
- Import: `import plotly.graph_objects as go` or `import plotly.express as px`
- Create interactive charts with hover details
- Use: `fig = px.line(...)` or `fig = go.Figure(...)`
- Update layout: `fig.update_layout(title='...', xaxis_title='...', yaxis_title='...')`
- Add formatting: `fig.update_traces(mode='lines+markers')`
- Final variable must be: `chart = fig`
- Save with: `chart.write_html('filename.html')` or `chart.write_image('filename.png')`
"""

SEABORN_INSTRUCTIONS = """
**Seaborn Instructions**:
- Import: `import seaborn as sns` and `import matplotlib.pyplot as plt`
- Set style: `sns.set_theme(style='whitegrid')`
- Use seaborn plot functions: `sns.lineplot()`, `sns.barplot()`, etc.
- Create figure: `fig, ax = plt.subplots(figsize=(12, 6))`
- Pass ax parameter: `sns.lineplot(data=data, x='...', y='...', ax=ax)`
- Customize with matplotlib: `ax.set_title()`, `ax.set_xlabel()`, etc.
- Final variable must be: `chart = fig`
"""
