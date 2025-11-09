-- ============================================================================
-- VIEW 1: fund_semantic_search_view
-- ============================================================================
-- Purpose: Optimized view for SEMANTIC SEARCH (vector/embedding-based search)
-- Use Case: Fuzzy queries like "Bradesco gold fund", "tech investing",
--           "sustainable funds", "Latin American equity"
--
-- This view combines all searchable text fields from multiple sources to create
-- a rich semantic search index. It will be used to generate embeddings for
-- natural language queries.
-- ============================================================================

CREATE OR REPLACE VIEW fund_semantic_search_view AS
WITH
-- Get the latest snapshot timestamp for each fund
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

-- Get active funds: Either marked ACTIVE OR in recent lamina filings
active_funds AS (
    SELECT
        f.fund_id.value AS fund_id,
        f.identifiers[1].value AS cnpj,
        f.legal_name,
        f.investment_class,
        f.anbima_classification,
        f.fund_type,
        f.structure,
        f.performance_benchmark,
        f.target_audience,
        f.manager_type,
        f.status,
        f.timestamp,
        -- Data quality score
        CASE
            WHEN f.status = 'ACTIVE' THEN 'HIGH'
            WHEN frl.cnpj IS NOT NULL THEN 'MEDIUM'  -- Unspecified but in lamina
            ELSE 'LOW'
        END AS data_quality
    FROM funds f
    INNER JOIN latest_fund_snapshots lfs
        ON f.fund_id.value = lfs.fund_id_value
        AND f.timestamp = lfs.latest_timestamp
    LEFT JOIN funds_in_recent_lamina frl
        ON f.identifiers[1].value = frl.cnpj
    WHERE f.identifiers[1].type = 'CNPJ'  -- Ensure we have CNPJ
        AND (
            f.status = 'ACTIVE'  -- Explicitly marked active
            OR (f.status = 'UNSPECIFIED' AND frl.cnpj IS NOT NULL)  -- Or unspecified with lamina filing
        )
),

-- Get lamina metadata (Portuguese descriptions, objectives, policies)
lamina_metadata AS (
    SELECT
        l.CNPJ_FUNDO_CLASSE AS cnpj,
        l.DENOM_SOCIAL AS legal_name_lamina,
        l.NM_FANTASIA AS trade_name,
        l.OBJETIVO AS objective,
        l.POLIT_INVEST AS investment_policy,
        l.PUBLICO_ALVO AS target_audience_desc,
        l.VL_PATRIM_LIQ AS net_asset_value,
        l.TAXA_ADM AS management_fee_pct,
        l.INVEST_INICIAL_MIN AS min_initial_investment,
        -- Get the most recent lamina data per fund
        ROW_NUMBER() OVER (PARTITION BY l.CNPJ_FUNDO_CLASSE ORDER BY l.DT_COMPTC DESC) AS rn
    FROM lamina_lamina_fi l
)

-- Final view combining all searchable text
SELECT
    af.fund_id,
    af.cnpj,
    af.legal_name,
    lm.trade_name,
    af.investment_class,
    af.anbima_classification,
    af.fund_type,
    af.structure,
    af.performance_benchmark,
    af.target_audience,
    lm.objective,
    lm.investment_policy,
    lm.target_audience_desc,
    af.manager_type,
    af.status AS original_status,
    af.data_quality,
    lm.net_asset_value,
    lm.management_fee_pct,
    lm.min_initial_investment,

    -- SEARCHABLE_TEXT: Concatenated text for embedding generation
    -- This combines all relevant text fields in Portuguese and English
    CONCAT_WS(' | ',
        af.legal_name,
        COALESCE(lm.trade_name, ''),
        COALESCE(af.investment_class, ''),
        COALESCE(af.anbima_classification, ''),
        COALESCE(lm.objective, ''),
        COALESCE(lm.investment_policy, ''),
        COALESCE(af.performance_benchmark, ''),
        COALESCE(af.manager_type, '')
    ) AS searchable_text,

    af.timestamp AS last_updated

FROM active_funds af
LEFT JOIN lamina_metadata lm
    ON af.cnpj = lm.cnpj
    AND lm.rn = 1  -- Only most recent lamina data

ORDER BY af.legal_name;


-- ============================================================================
-- USAGE EXAMPLES FOR SEMANTIC SEARCH VIEW
-- ============================================================================

-- Example 1: Preview the searchable text
-- SELECT cnpj, legal_name, LEFT(searchable_text, 200) as text_preview
-- FROM fund_semantic_search_view
-- LIMIT 5;

-- Example 2: Count total funds available for semantic search
-- SELECT COUNT(*) as total_active_funds
-- FROM fund_semantic_search_view;

-- Example 3: Check funds with richest metadata (longest searchable text)
-- SELECT cnpj, legal_name, LENGTH(searchable_text) as text_length
-- FROM fund_semantic_search_view
-- ORDER BY text_length DESC
-- LIMIT 10;

-- Example 4: Find funds by keyword (before implementing embeddings)
-- SELECT cnpj, legal_name, investment_class
-- FROM fund_semantic_search_view
-- WHERE LOWER(searchable_text) LIKE '%bradesco%'
--    OR LOWER(searchable_text) LIKE '%ouro%'  -- 'gold' in Portuguese
-- LIMIT 10;
