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
