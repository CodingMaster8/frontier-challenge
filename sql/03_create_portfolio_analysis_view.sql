-- ============================================================================
-- VIEW 3: fund_portfolio_analysis_view
-- ============================================================================
-- Purpose: Optimized view for PORTFOLIO-BASED SEARCH (Holdings analysis)
-- Use Case: Queries like:
--   - "Funds that invest in tech stocks"
--   - "Funds with USD exposure"
--   - "Funds holding Petrobras"
--   - "Funds investing in government bonds"
--   - "Funds with exposure to Latin American assets"
--
-- This view joins funds with their portfolio positions and asset details,
-- enabling searches based on what the funds actually hold.
-- ============================================================================

CREATE OR REPLACE VIEW fund_portfolio_analysis_view AS
WITH
-- Get the latest snapshot for each fund
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

-- Get active funds
active_funds AS (
    SELECT
        f.fund_id.value AS fund_id,
        f.identifiers[1].value AS cnpj,
        f.legal_name,
        f.investment_class,
        f.anbima_classification,
        f.fund_type,
        f.status,
        f.timestamp,
        -- Data quality score
        CASE
            WHEN f.status = 'ACTIVE' THEN 'HIGH'
            WHEN frl.cnpj IS NOT NULL THEN 'MEDIUM'
            ELSE 'LOW'
        END AS data_quality
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

-- Get latest positions for each fund
latest_positions AS (
    SELECT
        fund_id.value AS fund_id_value,
        MAX(timestamp) AS latest_position_date
    FROM positions
    GROUP BY fund_id.value
),

-- Join positions with asset details
portfolio_holdings AS (
    SELECT
        p.fund_id.value AS fund_id,
        p.asset_id.value AS asset_id,
        p.timestamp AS position_date,
        p.quantity,
        p.current_market_value.value AS market_value,
        p.current_market_value.currency AS position_currency,

        -- Calculate weight from cost_basis if available
        p.cost_basis.value AS cost_basis,

        -- Asset details
        a.asset_class,
        a.financial_instrument,
        a.financial_instrument_description,
        a.name AS asset_name,
        a.short_name AS asset_short_name,
        a.currency AS asset_currency,
        a.country AS asset_country,
        a.issuer.issuer_name AS issuer_name,
        a.issuer.issuer_type AS issuer_type,
        a.status AS asset_status

    FROM positions p
    INNER JOIN latest_positions lp
        ON p.fund_id.value = lp.fund_id_value
        AND p.timestamp = lp.latest_position_date
    INNER JOIN assets a
        ON p.asset_id.value = a.asset_id.value
        AND p.timestamp = a.timestamp
    WHERE a.status = 'ACTIVE'
),

-- Calculate portfolio weights (since weight column doesn't exist)
portfolio_with_weights AS (
    SELECT
        ph.*,
        ph.market_value / SUM(ph.market_value) OVER (PARTITION BY ph.fund_id) * 100 AS portfolio_weight_pct
    FROM portfolio_holdings ph
    WHERE ph.market_value IS NOT NULL
),

-- Aggregate portfolio composition by asset type
portfolio_composition_by_asset_class AS (
    SELECT
        pw.fund_id,
        pw.asset_class,
        COUNT(DISTINCT pw.asset_id) AS num_assets,
        SUM(pw.market_value) AS total_market_value,
        AVG(pw.portfolio_weight_pct) AS avg_weight_pct,
        SUM(pw.portfolio_weight_pct) AS total_weight_pct
    FROM portfolio_with_weights pw
    GROUP BY pw.fund_id, pw.asset_class
),

-- Aggregate by financial instrument
portfolio_composition_by_instrument AS (
    SELECT
        pw.fund_id,
        pw.financial_instrument,
        pw.financial_instrument_description,
        COUNT(DISTINCT pw.asset_id) AS num_assets,
        SUM(pw.market_value) AS total_market_value,
        SUM(pw.portfolio_weight_pct) AS total_weight_pct
    FROM portfolio_with_weights pw
    GROUP BY pw.fund_id, pw.financial_instrument, pw.financial_instrument_description
),

-- Aggregate by country exposure
portfolio_composition_by_country AS (
    SELECT
        pw.fund_id,
        pw.asset_country,
        COUNT(DISTINCT pw.asset_id) AS num_assets,
        SUM(pw.market_value) AS total_market_value,
        SUM(pw.portfolio_weight_pct) AS total_weight_pct
    FROM portfolio_with_weights pw
    WHERE pw.asset_country IS NOT NULL
    GROUP BY pw.fund_id, pw.asset_country
),

-- Aggregate by currency exposure
portfolio_composition_by_currency AS (
    SELECT
        pw.fund_id,
        pw.asset_currency,
        COUNT(DISTINCT pw.asset_id) AS num_assets,
        SUM(pw.market_value) AS total_market_value,
        SUM(pw.portfolio_weight_pct) AS total_weight_pct
    FROM portfolio_with_weights pw
    WHERE pw.asset_currency != 'UNKNOWN'
    GROUP BY pw.fund_id, pw.asset_currency
),

-- Get top holdings per fund
top_holdings AS (
    SELECT
        pw.fund_id,
        pw.asset_id,
        pw.asset_name,
        pw.asset_short_name,
        pw.asset_class,
        pw.financial_instrument,
        pw.portfolio_weight_pct,
        pw.market_value,
        pw.issuer_name,
        ROW_NUMBER() OVER (PARTITION BY pw.fund_id ORDER BY pw.portfolio_weight_pct DESC) AS holding_rank
    FROM portfolio_with_weights pw
)
-- Final view: Fund with portfolio composition summaries
SELECT
    af.fund_id,
    af.cnpj,
    af.legal_name,
    af.investment_class,
    af.anbima_classification,
    af.status AS original_status,
    af.data_quality,
    af.timestamp AS last_updated,

    -- Portfolio diversity metrics
    (SELECT COUNT(DISTINCT asset_id) FROM portfolio_with_weights pw2 WHERE pw2.fund_id = af.fund_id) AS total_positions,
    (SELECT COUNT(DISTINCT asset_class) FROM portfolio_with_weights pw2 WHERE pw2.fund_id = af.fund_id) AS num_asset_classes,
    (SELECT COUNT(DISTINCT asset_country) FROM portfolio_with_weights pw2 WHERE pw2.fund_id = af.fund_id AND asset_country IS NOT NULL) AS num_countries,

    -- Top 5 holdings (concatenated for quick preview)
    (SELECT STRING_AGG(asset_short_name || ' (' || ROUND(portfolio_weight_pct, 2) || '%)', ', ')
     FROM (SELECT * FROM top_holdings th WHERE th.fund_id = af.fund_id AND th.holding_rank <= 5 ORDER BY holding_rank)
    ) AS top_5_holdings,

    -- Asset class breakdown (as string for readability)
    (SELECT STRING_AGG(asset_class || ':' || ROUND(total_weight_pct, 2) || '%', ' | ' ORDER BY total_weight_pct DESC)
     FROM portfolio_composition_by_asset_class pcac
     WHERE pcac.fund_id = af.fund_id
    ) AS asset_class_breakdown,

    -- Country exposure breakdown (as string for readability)
    (SELECT STRING_AGG(asset_country || ':' || ROUND(total_weight_pct, 2) || '%', ' | ' ORDER BY total_weight_pct DESC)
     FROM portfolio_composition_by_country pcc
     WHERE pcc.fund_id = af.fund_id
    ) AS country_exposure,

    -- Currency exposure breakdown (as string for readability)
    (SELECT STRING_AGG(asset_currency || ':' || ROUND(total_weight_pct, 2) || '%', ' | ' ORDER BY total_weight_pct DESC)
     FROM portfolio_composition_by_currency pcur
     WHERE pcur.fund_id = af.fund_id
    ) AS currency_exposure,

    -- Instrument type breakdown (top 10, as string for readability)
    (SELECT STRING_AGG(financial_instrument_description || ':' || ROUND(total_weight_pct, 2) || '%', ' | ' ORDER BY total_weight_pct DESC)
     FROM (SELECT * FROM portfolio_composition_by_instrument pci WHERE pci.fund_id = af.fund_id ORDER BY total_weight_pct DESC LIMIT 10)
    ) AS instrument_breakdown,

    -- ========================================================================
    -- PIVOT COLUMNS: Asset Class Exposure (% allocation)
    -- ========================================================================
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'CASH'), 0) AS pct_cash,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'DERIVATIVES'), 0) AS pct_derivatives,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'EQUITY'), 0) AS pct_equity,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'FIXED_INCOME'), 0) AS pct_fixed_income,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'INVESTMENT_FUND'), 0) AS pct_investment_fund,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_asset_class pcac WHERE pcac.fund_id = af.fund_id AND pcac.asset_class = 'UNSPECIFIED'), 0) AS pct_unspecified,

    -- ========================================================================
    -- PIVOT COLUMNS: Country Exposure (% allocation) - Most Common
    -- ========================================================================
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'BRA'), 0) AS pct_bra,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'USA'), 0) AS pct_usa,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'LUX'), 0) AS pct_lux,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'IRL'), 0) AS pct_irl,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'CYM'), 0) AS pct_cym,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'GBR'), 0) AS pct_gbr,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'DEU'), 0) AS pct_deu,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'FRA'), 0) AS pct_fra,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'CHE'), 0) AS pct_che,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'NLD'), 0) AS pct_nld,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'JPN'), 0) AS pct_jpn,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'CAN'), 0) AS pct_can,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'ESP'), 0) AS pct_esp,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'ITA'), 0) AS pct_ita,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'AUS'), 0) AS pct_aus,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'MEX'), 0) AS pct_mex,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'ARG'), 0) AS pct_arg,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_country pcc WHERE pcc.fund_id = af.fund_id AND pcc.asset_country = 'CHL'), 0) AS pct_chl,

    -- ========================================================================
    -- PIVOT COLUMNS: Currency Exposure (% allocation)
    -- Note: Most positions are in BRL, but we can add others if needed
    -- ========================================================================
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_currency pcur WHERE pcur.fund_id = af.fund_id AND pcur.asset_currency = 'BRL'), 0) AS pct_brl,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_currency pcur WHERE pcur.fund_id = af.fund_id AND pcur.asset_currency = 'USD'), 0) AS pct_usd,
    COALESCE((SELECT ROUND(total_weight_pct, 2) FROM portfolio_composition_by_currency pcur WHERE pcur.fund_id = af.fund_id AND pcur.asset_currency = 'EUR'), 0) AS pct_eur

FROM active_funds af

ORDER BY af.legal_name;


-- ============================================================================
-- USAGE EXAMPLES FOR PORTFOLIO ANALYSIS VIEW
-- ============================================================================

-- ============================================================================
-- IMPORTANT: PIVOT COLUMNS FOR EASY AGENT QUERIES
-- ============================================================================
-- The view includes pivot columns for direct numeric filtering:
--
-- Asset Class Columns:
--   pct_cash, pct_derivatives, pct_equity, pct_fixed_income,
--   pct_investment_fund, pct_unspecified
--
-- Country Columns:
--   pct_bra, pct_usa, pct_lux, pct_irl, pct_cym, pct_gbr, pct_deu,
--   pct_fra, pct_che, pct_nld, pct_jpn, pct_can, pct_esp, pct_ita,
--   pct_aus, pct_mex, pct_arg, pct_chl
--
-- Currency Columns:
--   pct_brl, pct_usd, pct_eur
--
-- ============================================================================

-- Example 1: Find funds with significant equity exposure (SIMPLE!)
-- SELECT cnpj, legal_name, pct_equity, pct_fixed_income
-- FROM fund_portfolio_analysis_view
-- WHERE pct_equity >= 50
-- ORDER BY pct_equity DESC;

-- Example 2: Find funds investing in US assets (SIMPLE!)
-- SELECT cnpj, legal_name, pct_usa, pct_bra
-- FROM fund_portfolio_analysis_view
-- WHERE pct_usa >= 30
-- ORDER BY pct_usa DESC;

-- Example 3: Find funds with USD currency exposure (SIMPLE!)
-- SELECT cnpj, legal_name, pct_usd, pct_brl
-- FROM fund_portfolio_analysis_view
-- WHERE pct_usd >= 20
-- ORDER BY pct_usd DESC;

-- Example 4: Find well-diversified funds (many positions)
-- SELECT cnpj, legal_name, total_positions, num_asset_classes, num_countries
-- FROM fund_portfolio_analysis_view
-- WHERE total_positions > 50
--   AND num_asset_classes >= 3
-- ORDER BY total_positions DESC;

-- Example 5: Find funds holding a specific asset (detailed view)
-- SELECT cnpj, legal_name, asset_name, portfolio_weight_pct, position_value
-- FROM fund_holdings_detail_view
-- WHERE LOWER(asset_name) LIKE '%petrobras%'
--    OR LOWER(issuer_name) LIKE '%petrobras%'
-- ORDER BY portfolio_weight_pct DESC;

-- Example 6: Find funds heavily invested in derivatives
-- SELECT cnpj, legal_name, instrument_breakdown
-- FROM fund_portfolio_analysis_view
-- WHERE instrument_breakdown LIKE '%DERIVATIVE%'
--    OR instrument_breakdown LIKE '%FUTURE%'
--    OR instrument_breakdown LIKE '%OPTION%'
-- ORDER BY legal_name;

-- Example 7: Find funds investing in government bonds
-- SELECT DISTINCT fhd.cnpj, fhd.legal_name, fhd.asset_name, fhd.portfolio_weight_pct
-- FROM fund_holdings_detail_view fhd
-- WHERE fhd.financial_instrument = 'GOVERNMENT_BOND'
-- ORDER BY fhd.portfolio_weight_pct DESC
-- LIMIT 20;

-- Example 8: Find funds with international diversification (>20% foreign)
-- SELECT cnpj, legal_name, country_exposure, total_positions
-- FROM fund_portfolio_analysis_view
-- WHERE country_exposure IS NOT NULL
--   AND country_exposure NOT LIKE 'BRA:100%'
--   AND num_countries > 1
-- ORDER BY num_countries DESC;
