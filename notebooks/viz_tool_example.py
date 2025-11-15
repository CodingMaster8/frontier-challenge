"""
Financial Visualization Tool - Example Usage

This notebook demonstrates how to use the Financial Visualization Tool
for Brazilian fund analysis.
"""

import sys
sys.path.append('..')

import logging
import pandas as pd
import asyncio
from frontier_challenge.tools import FinancialVisualizationTool

# Configure logging to see execution time traces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example 1: Load sample fund data
print("📊 Loading sample fund data...")

# Create sample data
sample_data = pd.DataFrame({
    'cnpj': ['00.000.000/0001-01', '00.000.000/0001-02', '00.000.000/0001-03'] * 12,
    'fund_name': ['Fundo A', 'Fundo B', 'Fundo C'] * 12,
    'date': pd.date_range('2024-01-01', periods=36, freq='M').repeat(1).tolist()[:36],
    'return_mtd': [0.5, 1.2, -0.3, 0.8, 1.5, -0.5, 0.9, 1.1, 0.2] * 4,
    'return_ytd': [5.2, 8.5, 3.1, 6.8, 9.2, 4.5, 7.3, 8.9, 5.7] * 4,
    'aum': [1000000, 5000000, 2500000] * 12,
    'sharpe_ratio': [1.2, 1.8, 0.9] * 12,
    'volatility': [8.5, 12.3, 6.2] * 12,
    'category': ['Renda Fixa', 'Multimercado', 'Ações'] * 12,
})

print(f"✅ Loaded {len(sample_data)} rows")
print(sample_data.head())

# Example 2: Initialize the tool
print("\n🔧 Initializing FinancialVisualizationTool...")

viz_tool = FinancialVisualizationTool(
    library="seaborn",
    image_format="png",
    output_dir="./output_visualizations",
    language="english"
)

print("✅ Tool initialized")

# Example 3: Create a simple visualization
print("\n📈 Creating visualization...")

async def create_viz_example():
    results = await viz_tool.create_visualization(
        data=sample_data,
        query="Mostrar a evolução dos retornos mensais ao longo do tempo para cada fundo",
        number_visualizations=1
    )

    print(f"\n✅ Generated {len(results)} visualization(s)")

    for i, result in enumerate(results, 1):
        print(f"\n--- Visualization {i} ---")
        print(f"Image Path: {result.image_path}")
        print(f"Type: {result.visualization_type}")
        print(f"Description: {result.description}")
        print(f"\nGenerated Code:")
        print(result.python_code[:200] + "...")

# Run the async function
asyncio.run(create_viz_example())

print("\n✨ Example completed!")
