"""
Streamlit app for the Financial Agent.
"""

import streamlit as st
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging BEFORE importing other modules
from docs.challenge_docs.logging_config import setup_logging
setup_logging(level=logging.DEBUG)  # Use DEBUG for maximum detail

logger = logging.getLogger(__name__)

from frontier_challenge.agent.graph import get_financial_agent_graph
from frontier_challenge.agent.prompts import GREETING_TEMPLATES
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

logging.getLogger('PIL').setLevel(logging.INFO)

# Status messages in different languages
STATUS_MESSAGES = {
    "en": {
        "idle": "Idle",
        "ready": "Ready",
        "analyzing_query": "Analyzing your query...",
        "searching_funds": "Searching for funds...",
        "filtering_data": "Filtering fund data...",
        "processing_results": "Processing results...",
        "generating_response": "Generating response...",
        "error": "Error occurred",
    },
    "pt": {
        "idle": "Inativo",
        "ready": "Pronto",
        "analyzing_query": "Analisando sua pergunta...",
        "searching_funds": "Buscando fundos...",
        "filtering_data": "Filtrando dados de fundos...",
        "processing_results": "Processando resultados...",
        "generating_response": "Gerando resposta...",
        "error": "Ocorreu um erro",
    }
}

def get_status_message(status: str, language: str = "pt") -> str:
    """Get user-friendly status message."""
    return STATUS_MESSAGES.get(language, STATUS_MESSAGES["en"]).get(status, status)

def display_status(placeholder, message: str):
    """Display status message in a dark gray box."""
    with placeholder.container():
        st.markdown(
            f"""
            <div style="
                padding: 12px 16px;
                border-radius: 8px;
                background-color: #2b2b2b;
                border-left: 4px solid #4a4a4a;
                color: #e0e0e0;
                font-size: 14px;
                margin: 0;
                width: 100%;
                box-sizing: border-box;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

# Configure page
# Use the frontier.png logo as the page icon
logo_path = Path(__file__).parent / "frontier.png"
st.set_page_config(
    page_title="FundAI - Brazilian Funds Assistant",
    page_icon=str(logo_path),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "language" not in st.session_state:
    st.session_state.language = "pt"  # Default to Portuguese

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "idle"

if "agent_graph" not in st.session_state:
    with st.spinner("Initializing FundAI..."):
        logger.info("🚀 Initializing agent graph...")
        # Initialize the agent with Streamlit interface
        checkpointer = MemorySaver()
        st.session_state.agent_graph = get_financial_agent_graph(
            db_path="data/br_funds.db",
            model_name="gpt-4o-mini",
            checkpointer=checkpointer,
            session_state=st.session_state  # Pass session state
        )
        st.session_state.thread_id = "streamlit-session-1"
        logger.info("✅ Agent graph initialized successfully")

# Sidebar
with st.sidebar:
    # Display the frontier logo
    if logo_path.exists():
        st.image(str(logo_path), width=150)

    st.title("Frontier AI")
    st.markdown("### Brazilian Investment Funds Assistant")
    st.markdown("---")

    st.markdown("#### About")
    st.markdown(
        """
        FrontierAI helps you explore and analyze Brazilian investment funds using:
        - 🔍 Semantic search for fuzzy matching
        - 📊 Structured filtering for precise queries
        - 💬 Natural language conversation
        """
    )

    st.markdown("---")

    # Language selector
    language = st.selectbox(
        "Language / Idioma",
        ["English", "Português"],
        index=1  # Default to Portuguese
    )

    lang_code = "pt" if language == "Português" else "en"

    # Update language in session state
    if st.session_state.language != lang_code:
        st.session_state.language = lang_code
        st.session_state.initialized = False  # Reset to show greeting in new language

    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.initialized = False
        st.session_state.thread_id = f"streamlit-session-{len(st.session_state.messages)}"
        st.rerun()

avatar = str(logo_path) if logo_path.exists() else None

# Show initial greeting
if not st.session_state.initialized:
    greeting = GREETING_TEMPLATES.get(st.session_state.language, GREETING_TEMPLATES["en"])
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.initialized = True

# Main chat interface
st.title("Frontier AI")

# Display chat messages
for message in st.session_state.messages:
    msg_avatar = avatar if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=msg_avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Brazilian funds..." if st.session_state.language == "en" else "Pergunte-me sobre fundos brasileiros..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    logger.info(f"👤 User query: {prompt}")

    # Get agent response
    with st.chat_message("assistant", avatar=avatar):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            logger.info("Invoking agent graph...")

            # Invoke the agent graph asynchronously
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": 50
            }

            logger.debug(f"Config: {config}")
            logger.debug(f"User language: {st.session_state.language}")

            # Create an async task to stream updates
            async def run_agent():
                """Run agent and stream status updates."""
                last_status = "idle"

                # Stream through the graph execution
                async for event in st.session_state.agent_graph.astream(
                    {
                        "messages": [HumanMessage(content=prompt)],
                        "user_language": st.session_state.language
                    },
                    config=config
                ):
                    # Extract status from the event
                    if isinstance(event, dict):
                        for node_name, node_data in event.items():
                            if isinstance(node_data, dict) and "current_status" in node_data:
                                current_status = node_data["current_status"]
                                if current_status != last_status:
                                    last_status = current_status
                                    status_msg = get_status_message(current_status, st.session_state.language)
                                    display_status(status_placeholder, status_msg)
                                    st.session_state.agent_status = current_status
                                    logger.info(f"📍 Status: {current_status}")

                # Get final result
                final_state = await st.session_state.agent_graph.aget_state(config)
                return final_state.values if final_state else None

            # Run the agent with status updates
            result = asyncio.run(run_agent())

            # Clear status placeholder
            status_placeholder.empty()

            logger.info("✅ Agent graph execution completed")
            logger.debug(f"Result keys: {result.keys() if result else 'None'}")

            # Extract the final response from the result
            if result and "messages" in result and len(result["messages"]) > 0:
                # Get the last AI message
                last_message = result["messages"][-1]
                full_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                logger.info(f"💬 Agent response: {full_response[:100]}...")
            else:
                full_response = "I apologize, but I couldn't generate a response. Please try again." if st.session_state.language == "en" else "Desculpe, não consegui gerar uma resposta. Tente novamente."
                logger.warning("⚠️ No valid response from agent")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"❌ Error during agent execution: {str(e)}", exc_info=True)
            status_placeholder.empty()
            error_msg = f"Error: {str(e)}" if st.session_state.language == "en" else f"Erro: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Powered by FrontierAI Brasil Team | Data: CVM Brazilian Funds</small>
    </div>
    """,
    unsafe_allow_html=True
)
