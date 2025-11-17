# Agent Evaluation Framework

Production-grade evaluation suite for the Financial Agent and its tools. This framework provides comprehensive testing and quality metrics for all agent components.

##  Overview

This evaluation framework quantifies the accuracy, performance, and robustness of:

- **Semantic Search Tool** (Pinecone + OpenAI embeddings)
- **Structured Filter Tool** (NL2SQL with LangChain)
- **Holdings Search Tool** (Fuzzy matching with Levenshtein distance)
- **Agent Routing** (Tool selection logic)

##  Architecture

```
evals/
├── __init__.py                 # Main exports
├── models.py                   # Data models and types
├── metrics.py                  # Evaluation metrics
├── evaluator.py                # Core evaluation framework
├── test_cases/                 # Test case definitions
│   ├── __init__.py
│   ├── semantic_search_cases.py
│   ├── structured_filter_cases.py
│   ├── holdings_search_cases.py
│   └── agent_routing_cases.py
├── run_semantic_eval.py        # Semantic search evaluator
├── run_sql_eval.py             # SQL filter evaluator
├── run_holdings_eval.py        # Holdings search evaluator
├── run_all_evals.py            # Comprehensive evaluator
└── results/                    # Evaluation results (generated)
```

##  Metrics

### Accuracy Metrics
- **Accuracy**: Ratio of correct predictions/results
- **Precision**: Ratio of relevant results in returned set
- **Recall**: Ratio of relevant results that were found
- **F1 Score**: Harmonic mean of precision and recall

### Performance Metrics
- **Latency**: Execution time in milliseconds
- **Error Rate**: Ratio of failed executions

### Quality Metrics
- **Relevance**: How relevant results are to the query
- **Completeness**: Whether all required fields are present
- **Robustness**: How well tools handle edge cases

## Test Cases

### Semantic Search Tool
- **Basic Cases** (18 total): Company names, investment types, ESG queries, risk profiles
- **Edge Cases** (8): Misspellings, mixed languages, very long queries, special characters
- **Performance Cases**: Top K variations, exact matches

### Structured Filter Tool (SQL)
- **Basic Cases** (10): Return filters, fee filters, NAV filters, risk filters
- **Comparison Cases** (3): Greater than, less than, range filters
- **Performance Cases** (2): 12-month returns, volatility
- **Edge Cases** (8): Conflicting criteria, NULL handling, extreme values, negative values
- **Security Cases** (2): SQL injection prevention
- **Sorting Cases** (2): Single and multi-column sorting
- **Classification Cases** (2): Investment class filters

### Holdings Search Tool
- **Basic Cases** (5): Company searches (Petrobras, Vale, Itau, banks, bonds)
- **Fuzzy Matching** (3): Misspellings, partial names, abbreviations
- **Grouping Cases** (2): Group by fund, top holdings
- **Edge Cases** (8): Non-existent companies, generic terms, multiple companies
- **Natural Language** (3): Complex queries in English and Portuguese
- **Weight Filtering** (2): Minimum weight, large positions
- **Sector Analysis** (2): Energy and financial sector exposure

### Agent Routing
- **Routing Cases** (13): Tests the agent's ability to select the correct tool
- **Ambiguous Cases** (2): Queries that could use multiple tools
- **Multi-Tool Cases** (2): Queries requiring multiple tools in sequence

## Usage

### Run Individual Tool Evaluations

```bash
# Evaluate Semantic Search Tool
python -m evals.run_semantic_eval

# Evaluate Structured Filter Tool
python -m evals.run_sql_eval

# Evaluate Holdings Search Tool
python -m evals.run_holdings_eval
```

### Run Comprehensive Evaluation

```bash
# Evaluate all tools at once
python -m evals.run_all_evals
```

This will:
1. Run all test suites for all tools
2. Generate detailed metrics for each tool
3. Produce overall summary statistics
4. Export results to JSON
5. Generate an HTML report

### Output Files

Results are saved in `evals/results/`:
- `semantic_search_eval.json` - Semantic search results
- `structured_filter_eval.json` - SQL filter results
- `holdings_search_eval.json` - Holdings search results
- `comprehensive_eval_YYYYMMDD_HHMMSS.json` - All results
- `eval_report_YYYYMMDD_HHMMSS.html` - HTML report

##  Interpreting Results

### Pass/Fail Thresholds

```python
PASS_RATE_THRESHOLD = 0.75      # 75% of cases must pass
ERROR_RATE_THRESHOLD = 0.15     # < 15% error rate
ACCURACY_THRESHOLD = 0.70       # > 70% accuracy
```

### Tool-Specific Thresholds

- **Semantic Search**: 80% pass rate (more strict)
- **Structured Filter**: 75% pass rate (more lenient due to SQL complexity)
- **Holdings Search**: 80% pass rate

### Example Output

```
COMPREHENSIVE EVALUATION SUMMARY
================================================================================

Total Cases: 85
Passed: 68 (80.0%)
Failed: 12 (14.1%)
Errors: 5 (5.9%)
Error Rate: 5.9%

Total Time: 45230.50ms
Avg Time per Case: 532.12ms
Overall Accuracy: 82.3%

PER-TOOL BREAKDOWN
--------------------------------------------------------------------------------

SEMANTIC_SEARCH
  Cases: 26
  Pass Rate: 22/26 (84.6%)
  Accuracy: 85.2%
  Precision: 88.5%
  Recall: 81.3%
  F1 Score: 84.8%
  Avg Latency: 245.32ms

STRUCTURED_FILTER
  Cases: 32
  Pass Rate: 24/32 (75.0%)
  Accuracy: 78.9%
  Precision: 82.1%
  Recall: 75.4%
  F1 Score: 78.6%
  Avg Latency: 892.45ms

HOLDINGS_SEARCH
  Cases: 27
  Pass Rate: 22/27 (81.5%)
  Accuracy: 83.7%
  Precision: 87.2%
  Recall: 79.8%
  F1 Score: 83.3%
  Avg Latency: 421.18ms
```

##  Adding New Test Cases

### 1. Create a Test Case

```python
from evals.models import EvalCase, ToolType

new_case = EvalCase(
    id="sem_new_001",
    name="Test case name",
    description="What this test case validates",
    tool_type=ToolType.SEMANTIC_SEARCH,
    input_query="Your test query",
    min_results=5,  # Optional: minimum expected results
    expected_funds=["CNPJ1", "CNPJ2"],  # Optional: expected CNPJs
    edge_case=False,  # Set True for edge cases
    tags=["category", "subcategory"],
)
```

### 2. Add to Test Suite

Add your case to the appropriate file in `evals/test_cases/`:
- `semantic_search_cases.py`
- `structured_filter_cases.py`
- `holdings_search_cases.py`
- `agent_routing_cases.py`

### 3. Run Evaluation

```bash
python -m evals.run_semantic_eval  # Or appropriate evaluator
```

##  Custom Metrics

You can add custom metrics by extending the `Metric` base class:

```python
from evals.metrics import Metric

class CustomMetric(Metric):
    @property
    def name(self) -> str:
        return "custom_metric"

    def compute(self, actual, expected, metadata=None) -> float:
        # Your metric computation logic
        return score
```

Then add it to the evaluator:

```python
evaluator = AgentEvaluator()
evaluator.metrics.append(CustomMetric())
```

##  Debugging Failed Tests

### View Failed Cases

```python
# Results are exported to JSON with full details
import json

with open('evals/results/semantic_search_eval.json', 'r') as f:
    results = json.load(f)

failed = [r for r in results if r['status'] == 'failed']
for result in failed:
    print(f"Case: {result['case_id']}")
    print(f"Error: {result['error_message']}")
    print(f"Failed checks: {result['failed_checks']}")
```

### Run Specific Test Cases

```python
from evals.test_cases import get_cases_by_tags

# Get only edge cases
edge_cases = get_cases_by_tags(['edge_case'])

# Get specific category
fuzzy_cases = get_cases_by_tags(['fuzzy'])
```

##  Continuous Integration

### GitHub Actions Example

```yaml
name: Agent Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.14'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
        run: |
          python -m evals.run_all_evals
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: eval-results
          path: evals/results/
```
