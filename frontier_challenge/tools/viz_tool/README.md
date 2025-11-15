# Financial Visualization Tool

Automatic visualization generation for Brazilian fund analysis, inspired by KViz but specialized for financial data.

## Overview

The Financial Visualization Tool uses a multi-stage LangGraph workflow to automatically generate insightful visualizations from DataFrames and natural language queries. It's specifically designed for Brazilian investment fund analysis.

## Architecture

The tool follows a **4-stage pipeline**:

```
┌─────────────────┐
│  1. Summarize   │  Analyze dataset structure, statistics, and quality
│     Dataset     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Propose     │  Generate N visualization proposals based on data
│  Visualizations │  and user query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Generate    │  Create Python code for each visualization
│     Code        │  (parallel execution for multiple visualizations)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Execute &   │  Run code, save charts, retry on errors
│     Save        │
└─────────────────┘
```

## Features

- 🤖 **Automated Analysis**: Understands your data structure automatically
- 💡 **Smart Proposals**: Suggests relevant visualizations based on financial domain expertise
- 🔄 **Error Recovery**: Automatically retries failed visualizations with corrections
- 📊 **Multiple Libraries**: Supports matplotlib, seaborn, and plotly
- 🌍 **Multilingual**: Supports English and Portuguese labels
- 💰 **Financial Focus**: Optimized for fund performance, risk metrics, and portfolio analysis

## Comparison with KViz

| Feature | KViz | Financial Viz Tool |
|---------|------|-------------------|
| **Domain** | General purpose | Brazilian funds |
| **Storage** | AWS S3 | Local filesystem |
| **LLMs** | Bedrock + OpenAI | OpenAI only |
| **Focus** | Generic datasets | Financial metrics |
| **Context** | Generic | Fund analysis specific |

## Installation

The tool is already part of the `frontier_challenge` package:

```python
from frontier_challenge.tools import FinancialVisualizationTool
```

## Usage

### Basic Example

```python
import pandas as pd
from frontier_challenge.tools import FinancialVisualizationTool

# Load your fund data
fund_df = pd.read_csv("fund_data.csv")

# Initialize the tool
viz_tool = FinancialVisualizationTool(
    library="matplotlib",
    image_format="png",
    output_dir="./visualizations",
    language="portuguese"
)

# Create visualizations
results = await viz_tool.create_visualization(
    data=fund_df,
    query="Mostrar os fundos com melhor retorno no último ano",
    number_visualizations=2
)

# Access results
for result in results:
    print(f"Image: {result.image_path}")
    print(f"Type: {result.visualization_type}")
    print(f"Description: {result.description}")
    print(f"Code:\n{result.python_code}")
```

### Synchronous Usage

```python
# For non-async contexts
results = viz_tool.create_visualization_sync(
    data=fund_df,
    query="Compare fund performance by category"
)
```

### Advanced Example

```python
# Custom configuration per request
results = await viz_tool.create_visualization(
    data=fund_df,
    query="Show risk vs return scatter plot for equity funds",
    number_visualizations=1,
    library="seaborn",  # Override default
    image_format="svg",  # Override default
)
```

## Query Examples

### Portuguese Queries
```python
# Performance Analysis
"Mostrar a evolução do patrimônio dos fundos ao longo do tempo"
"Comparar o retorno YTD dos principais fundos multimercado"
"Exibir a distribuição de retornos mensais por categoria ANBIMA"

# Risk Analysis
"Gráfico de volatilidade vs retorno dos fundos"
"Mostrar os fundos com melhor índice de Sharpe"
"Comparar o VaR dos fundos de renda fixa"

# Fee Analysis
"Impacto das taxas de administração no retorno líquido"
"Comparar taxa de performance entre gestoras"
"Ranking de fundos por custo total"

# Portfolio Composition
"Composição do portfólio dos maiores fundos"
"Distribuição de ativos por classe"
"Concentração setorial dos fundos de ações"
```

### English Queries
```python
# Performance
"Show fund performance over the last 12 months"
"Compare returns by fund category"
"Top 10 performing funds this year"

# Risk-Return
"Risk-return scatter plot"
"Sharpe ratio comparison across fund types"
"Drawdown analysis for equity funds"
```

## Configuration

### Visualization Libraries

**Matplotlib** (default)
- Best for: Static publication-quality charts
- Format: PNG, SVG, PDF
- Style: Professional, clean

**Seaborn**
- Best for: Statistical visualizations
- Format: PNG, SVG, PDF
- Style: Modern, colorful

**Plotly**
- Best for: Interactive dashboards
- Format: HTML, PNG
- Style: Interactive, web-ready

### Output Formats

- `png`: Standard raster format (default)
- `svg`: Scalable vector graphics
- `html`: Interactive (plotly only)
- `pdf`: Print-ready

## Data Requirements

The tool works best with DataFrames containing:

### Essential Columns
- Fund identifiers (CNPJ, name)
- Dates or periods
- Performance metrics (returns, AUM)

### Recommended Columns
- Risk metrics (volatility, Sharpe, VaR)
- Fees (admin, performance)
- Categories (ANBIMA class)
- Benchmark data

### Example Schema
```python
fund_df = pd.DataFrame({
    'cnpj': [...],
    'fund_name': [...],
    'date': [...],
    'return_mtd': [...],
    'return_ytd': [...],
    'aum': [...],
    'sharpe_ratio': [...],
    'volatility': [...],
    'category': [...],
})
```

## Workflow Details

### Stage 1: Data Summarization
- Analyzes column types and statistics
- Identifies financial metrics
- Detects data quality issues
- Extracts key relationships

### Stage 2: Visualization Proposals
- Generates N visualization ideas
- Matches chart types to data
- Considers financial best practices
- Provides rationale for each proposal

### Stage 3: Code Generation
- Creates Python code for each visualization
- Handles data transformations
- Applies financial styling
- Includes proper labels and formatting

### Stage 4: Execution & Error Recovery
- Executes generated code safely
- Saves charts to disk
- Retries on errors (up to 3 times)
- Returns metadata with results

## Output Structure

Each visualization returns a `VisualizationResult`:

```python
@dataclass
class VisualizationResult:
    image_path: str          # "/tmp/abc123.png"
    python_code: str         # The generated code
    description: str         # What the viz shows
    visualization_type: str  # "line chart", "bar chart", etc.
```

## Integration with Agent

To integrate with the financial agent:

```python
from frontier_challenge.agent import get_financial_agent_graph
from frontier_challenge.tools import FinancialVisualizationTool

# Add as a tool to the agent
viz_tool = FinancialVisualizationTool()

# Use in agent workflow
# (Integration code depends on your agent architecture)
```

## Best Practices

### 1. Data Preparation
```python
# Clean data before visualization
fund_df = fund_df.dropna(subset=['return_ytd'])
fund_df['date'] = pd.to_datetime(fund_df['date'])
fund_df = fund_df.sort_values('date')
```

### 2. Clear Queries
```python
# Good: Specific and actionable
query = "Show monthly returns for equity funds in 2024"

# Less ideal: Too vague
query = "Show me something interesting"
```

### 3. Reasonable Limits
```python
# Generate 1-3 visualizations per request
# More than 3 increases latency significantly
results = await viz_tool.create_visualization(
    data=fund_df,
    query=query,
    number_visualizations=2  # Sweet spot
)
```

### 4. Error Handling
```python
try:
    results = await viz_tool.create_visualization(
        data=fund_df,
        query=query
    )
except Exception as e:
    logger.error(f"Visualization failed: {e}")
    # Fallback to manual plotting
```

## Limitations

1. **Code Execution**: Runs arbitrary code - use only in controlled environments
2. **Performance**: Each visualization takes 10-30 seconds to generate
3. **Data Size**: Works best with DataFrames < 100k rows
4. **Dependencies**: Requires matplotlib/seaborn/plotly installed
5. **LLM Costs**: Uses OpenAI models (gpt-4o, o3-mini)

## Troubleshooting

### Common Issues

**Issue**: "LLM must be provided"
```python
# Solution: Check OpenAI API key
import os
print(os.getenv("OPENAI_API_KEY"))
```

**Issue**: Code execution fails
```python
# Solution: Check data types and missing values
print(fund_df.info())
print(fund_df.isnull().sum())
```

**Issue**: Poor quality visualizations
```python
# Solution: Provide more context
results = await viz_tool.create_visualization(
    data=fund_df,
    query="Show equity fund returns. Focus on top 10 by AUM. Compare vs CDI benchmark.",
    number_visualizations=1
)
```

## Development

### Running Tests
```bash
# Unit tests
pytest tests/test_viz_tool.py

# Integration tests
pytest tests/test_viz_tool_integration.py
```

### Adding New Libraries
To add support for a new library (e.g., Altair):

1. Add template in `utils.py`:
```python
def get_chart_template(library: str):
    # ...
    elif library == "altair":
        template = "..."
        instructions = "..."
        return template, instructions
```

2. Update `save_chart()` in `utils.py`
3. Add tests

## Roadmap

- [ ] Support for Altair/Vega-Lite
- [ ] Async batch processing
- [ ] Chart caching
- [ ] Custom styling themes
- [ ] Export to PowerPoint/PDF reports
- [ ] Integration with Streamlit app

## References

- **KViz**: Original inspiration - [GitHub](https://github.com/kuona/kviz)
- **LangGraph**: Workflow framework - [Docs](https://langchain-ai.github.io/langgraph/)
- **Matplotlib**: Visualization library - [Docs](https://matplotlib.org/)
- **Seaborn**: Statistical graphics - [Docs](https://seaborn.pydata.org/)

## License

Same as parent project (frontier-challenge).
