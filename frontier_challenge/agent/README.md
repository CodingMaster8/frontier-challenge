# Agent Module

Core agent implementation using LangGraph for conversational AI orchestration.

## Purpose

This module implements the main conversational agent that orchestrates tool usage, manages conversation state, and provides natural language interaction with the fund database.

## Components

### graph.py
Main LangGraph state machine that defines the agent's behavior flow:
- Tool routing based on user queries
- Conversational greeting and error handling
- Tool execution and result formatting
- Bilingual response generation
- Visualization decision logic

### models.py
Pydantic models for agent state management:
- `AgentState` - Core conversation state with messages and internal reasoning
- `ToolReasoningResponse` - Structured tool selection decisions
- `VisualizationDecisionResponse` - Visualization generation decisions

### prompts.py
System prompts for agent behavior:
- `FINANCIAL_AGENT_SYSTEM_PROMPT` - Main agent personality and capabilities
- `TOOL_ROUTER_SYSTEM_PROMPT` - Tool selection logic
- `GREETING_TEMPLATES` - Multilingual greetings
- `VISUALIZATION_DECISION_PROMPT` - Visualization generation logic

### utils.py
Helper functions for message processing, date handling, language detection, and result formatting.

## Design Decisions

**State-Based Architecture**: Uses LangGraph's state management instead of traditional chains to enable complex conversation flows with multiple decision points.

**Internal Monologue**: Maintains separate message streams for user-facing responses and internal reasoning to improve debugging and system transparency.

**Tool Routing**: Implements explicit tool routing logic using LLM to classify queries and select appropriate tools, avoiding expensive vector similarity searches for every query.

**Error Recovery**: Includes retry logic and error guidance to handle tool failures gracefully without breaking the conversation flow.

**Bilingual Support**: Automatically detects user language preference and maintains consistent language throughout the conversation.

## Graph Flow

1. **Router Node**: Analyzes user query to determine if tools are needed
2. **Greeting Node**: Handles initial greetings without tool invocation
3. **Tool Execution Node**: Invokes selected tools and collects results
4. **Formatting Node**: Processes tool results into structured formats
5. **Response Node**: Generates final natural language response
6. **Visualization Node**: Optionally generates charts based on results
