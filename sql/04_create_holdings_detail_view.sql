-- ============================================================================
-- VIEW 4: fund_holdings_detail_view
-- ============================================================================
-- Purpose: Detailed asset-level holdings for deep-dive portfolio queries
-- Use Case: Queries that need individual position details like:
--   - "Show me all positions in Petrobras across all funds"
--   - "Which funds hold more than 5% in a specific asset?"
--   - "List all government bonds held by Fund X"
--
-- This view provides granular, position-by-position data for advanced analysis.
-- ============================================================================

CREATE OR REPLACE VIEW fund_holdings_detail_view AS
WITH
latest_fund_snapshots AS (
    SELECT
        fund_id.value AS fund_id_value,
        MAX(timestamp) AS latest_timestamp
    FROM funds
    GROUP BY fund_id.value
),

-- Get funds that have recent lamina filings (proof they're active)
-- Using 12-month lookback from current date (Nov 2025 - 12 months = Nov 2024)
funds_in_recent_lamina AS (
    SELECT DISTINCT
        CNPJ_FUNDO_CLASSE AS cnpj,
        MAX(DT_COMPTC) AS last_filing_date
    FROM lamina_lamina_fi
    WHERE DT_COMPTC >= '2024-11-01'  -- Last 12 months of filings
    GROUP BY CNPJ_FUNDO_CLASSE
),

active_funds AS (
    SELECT
        f.fund_id.value AS fund_id,
        f.identifiers[1].value AS cnpj,
        f.legal_name,
        f.investment_class
    FROM funds f
    INNER JOIN latest_fund_snapshots lfs
        ON f.fund_id.value = lfs.fund_id_value
        AND f.timestamp = lfs.latest_timestamp
    LEFT JOIN funds_in_recent_lamina frl
        ON f.identifiers[1].value = frl.cnpj
    WHERE f.identifiers[1].type = 'CNPJ'
        AND (
            f.status = 'ACTIVE'
            OR (f.status = 'UNSPECIFIED' AND frl.cnpj IS NOT NULL)
        )
),

latest_positions AS (
    SELECT
        fund_id.value AS fund_id_value,
        MAX(timestamp) AS latest_position_date
    FROM positions
    GROUP BY fund_id.value
)

SELECT
    af.fund_id,
    af.cnpj,
    af.legal_name,
    af.investment_class,

    p.position_id.value AS position_id,
    p.asset_id.value AS asset_id,
    p.quantity,
    p.current_market_value.value AS position_value,
    p.current_market_value.currency AS position_currency,
    p.current_market_value.value / SUM(p.current_market_value.value) OVER (PARTITION BY af.fund_id) * 100 AS portfolio_weight_pct,

    -- Asset information
    a.asset_class,
    a.financial_instrument,
    a.financial_instrument_description,
    a.name AS asset_name,
    a.short_name AS asset_short_name,
    a.currency AS asset_currency,
    a.country AS asset_country,
    a.issuer.issuer_name AS issuer_name,
    a.issuer.issuer_type AS issuer_type,

    p.timestamp AS position_date

FROM active_funds af
INNER JOIN positions p ON af.fund_id = p.fund_id.value
INNER JOIN latest_positions lp
    ON p.fund_id.value = lp.fund_id_value
    AND p.timestamp = lp.latest_position_date
INNER JOIN assets a
    ON p.asset_id.value = a.asset_id.value
    AND p.timestamp = a.timestamp
WHERE a.status = 'ACTIVE'

ORDER BY af.legal_name, portfolio_weight_pct DESC;


-- ============================================================================
-- USAGE EXAMPLES FOR HOLDINGS DETAIL VIEW
-- ============================================================================

-- Example 1: Find all funds holding a specific asset
-- SELECT cnpj, legal_name, asset_name, portfolio_weight_pct, position_value
-- FROM fund_holdings_detail_view
-- WHERE LOWER(asset_name) LIKE '%petrobras%'
--    OR LOWER(issuer_name) LIKE '%petrobras%'
-- ORDER BY portfolio_weight_pct DESC;

-- Example 2: Find all government bond holdings
-- SELECT DISTINCT cnpj, legal_name, asset_name, portfolio_weight_pct
-- FROM fund_holdings_detail_view
-- WHERE financial_instrument = 'GOVERNMENT_BOND'
-- ORDER BY portfolio_weight_pct DESC
-- LIMIT 20;

-- Example 3: Find concentrated positions (>10% of portfolio)
-- SELECT cnpj, legal_name, asset_name, portfolio_weight_pct
-- FROM fund_holdings_detail_view
-- WHERE portfolio_weight_pct > 10
-- ORDER BY portfolio_weight_pct DESC;

-- Example 4: Analyze holdings by issuer
-- SELECT issuer_name, COUNT(DISTINCT fund_id) as num_funds,
--        SUM(position_value) as total_exposure
-- FROM fund_holdings_detail_view
-- WHERE issuer_name IS NOT NULL
-- GROUP BY issuer_name
-- ORDER BY total_exposure DESC
-- LIMIT 20;
