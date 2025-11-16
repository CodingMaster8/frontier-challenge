-- ============================================================================
-- MASTER SCRIPT: Create All Fund Search Views
-- ============================================================================
-- This script creates all four specialized views for the hybrid search system
--
-- Execution order:
-- 1. Semantic Search View - for embedding-based fuzzy search
-- 2. Structured Filter View - for precise SQL filtering
-- 3. Portfolio Analysis View - for holdings-based search
-- 4. Holdings Detail View - for detailed position-level queries
--
-- Usage: Run this entire script in DuckDB to set up all views
-- ============================================================================

\echo 'Creating fund search views...'
\echo ''

-- ============================================================================
-- VIEW 1: SEMANTIC SEARCH VIEW
-- ============================================================================
\echo 'Creating fund_semantic_search_view...'

\ir 01_create_semantic_search_view.sql

\echo '✓ fund_semantic_search_view created'
\echo ''

-- ============================================================================
-- VIEW 2: STRUCTURED FILTER VIEW
-- ============================================================================
\echo 'Creating fund_structured_filter_view...'

\ir 02_create_structured_filter_view.sql

\echo '✓ fund_structured_filter_view created'
\echo ''

-- ============================================================================
-- VIEW 3: PORTFOLIO ANALYSIS VIEW
-- ============================================================================
\echo 'Creating fund_portfolio_analysis_view...'

\ir 03_create_portfolio_analysis_view.sql

\echo '✓ fund_portfolio_analysis_view created'
\echo ''

-- ============================================================================
-- VIEW 4: HOLDINGS DETAIL VIEW
-- ============================================================================
\echo 'Creating fund_holdings_detail_view...'

\ir 04_create_holdings_detail_view.sql

\echo '✓ fund_holdings_detail_view created'
\echo ''

-- ============================================================================
-- VERIFY VIEWS WERE CREATED
-- ============================================================================
\echo 'Verifying views...'
\echo ''

SELECT 'VIEWS CREATED:' AS status;
SELECT table_name
FROM information_schema.tables
WHERE table_type = 'VIEW'
  AND table_name LIKE 'fund_%'
ORDER BY table_name;

\echo ''
\echo '============================================'
\echo 'All views created successfully! 🎉'
\echo '============================================'
\echo ''
\echo 'Quick Stats:'

SELECT
    'fund_semantic_search_view' AS view_name,
    COUNT(*) AS total_funds
FROM fund_semantic_search_view

UNION ALL

SELECT
    'fund_structured_filter_view' AS view_name,
    COUNT(*) AS total_funds
FROM fund_structured_filter_view

UNION ALL

SELECT
    'fund_portfolio_analysis_view' AS view_name,
    COUNT(*) AS total_funds
FROM fund_portfolio_analysis_view

UNION ALL

SELECT
    'fund_holdings_detail_view' AS view_name,
    COUNT(*) AS total_positions
FROM fund_holdings_detail_view;
