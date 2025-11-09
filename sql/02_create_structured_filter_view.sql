-- ============================================================================
-- VIEW 2: fund_structured_filter_view
-- ============================================================================
-- Purpose: Optimized view for STRUCTURED FILTERING (Text-to-SQL queries)
-- Use Case: Precise queries like:
--   - "funds with >15% YTD returns"
--   - "equity funds with minimum investment under R$1000"
--   - "funds that beat CDI benchmark"
--   - "multimercado funds with low fees"
--
-- This view provides all numeric and categorical fields needed for filtering,
-- sorting, and ranking funds based on specific criteria.
-- ============================================================================

CREATE OR REPLACE VIEW fund_structured_filter_view AS
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

-- Get active funds with classification metadata
active_funds AS (
    SELECT
        f.fund_id.value AS fund_id,
        f.identifiers[1].value AS cnpj,
        f.legal_name,
        f.investment_class,
        f.anbima_classification,
        f.fund_type,
        f.structure,
        f.status,
        f.performance_benchmark,
        f.target_audience,
        f.registration_date,
        f.constitution_date,
        f.activity_start_date,
        f.manager_type,
        f.net_asset_value.value AS nav_from_funds,
        f.net_asset_value.currency AS nav_currency,
        f.management_fee AS management_fee_from_funds,
        f.performance_fee AS performance_fee_from_funds,
        f.is_fund_of_funds,
        f.is_exclusive_fund,
        f.can_invest_abroad_100_pct,
        f.has_long_term_taxation,
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

-- Get financial metrics from lamina
lamina_metrics AS (
    SELECT
        l.CNPJ_FUNDO_CLASSE AS cnpj,
        l.DENOM_SOCIAL AS legal_name_lamina,
        l.NM_FANTASIA AS trade_name,

        -- Investment constraints
        TRY_CAST(l.INVEST_INICIAL_MIN AS DOUBLE) AS min_initial_investment,
        TRY_CAST(l.INVEST_ADIC AS DOUBLE) AS min_additional_investment,
        TRY_CAST(l.RESGATE_MIN AS DOUBLE) AS min_redemption,
        TRY_CAST(l.VL_MIN_PERMAN AS DOUBLE) AS min_balance_required,
        TRY_CAST(l.QT_DIA_CAREN AS INTEGER) AS lockup_days,

        -- Fees
        TRY_CAST(l.TAXA_ADM AS DOUBLE) AS management_fee_pct,
        TRY_CAST(l.TAXA_PERFM AS DOUBLE) AS performance_fee_pct,
        TRY_CAST(l.TAXA_ENTR AS DOUBLE) AS entry_fee_pct,
        TRY_CAST(l.TAXA_SAIDA AS DOUBLE) AS exit_fee_pct,
        TRY_CAST(l.PR_PL_DESPESA AS DOUBLE) AS expense_ratio_pct,

        -- Fund size and risk
        TRY_CAST(l.VL_PATRIM_LIQ AS DOUBLE) AS net_asset_value,
        l.CLASSE_RISCO_ADMIN AS risk_class,

        -- Performance metrics (5-year)
        TRY_CAST(l.PR_RENTAB_FUNDO_5ANO AS DOUBLE) AS return_5y_pct,
        TRY_CAST(l.PR_VARIACAO_INDICE_REFER_5ANO AS DOUBLE) AS benchmark_return_5y_pct,
        TRY_CAST(l.QT_ANO_PERDA AS INTEGER) AS years_with_losses,

        -- Portfolio constraints
        TRY_CAST(l.PR_PL_ATIVO_EXTERIOR AS DOUBLE) AS foreign_assets_pct,
        TRY_CAST(l.PR_PL_ATIVO_CRED_PRIV AS DOUBLE) AS private_credit_pct,
        TRY_CAST(l.PR_PL_ALAVANC AS DOUBLE) AS leverage_ratio,

        -- Liquidity
        TRY_CAST(l.QT_DIA_CONVERSAO_COTA_COMPRA AS INTEGER) AS purchase_settlement_days,
        TRY_CAST(l.QT_DIA_CONVERSAO_COTA_RESGATE AS INTEGER) AS redemption_settlement_days,
        TRY_CAST(l.QT_DIA_PAGTO_RESGATE AS INTEGER) AS redemption_payment_days,

        l.PUBLICO_ALVO AS target_audience_desc,
        l.INDICE_REFER AS benchmark_index,
        l.DT_COMPTC AS lamina_report_date,

        ROW_NUMBER() OVER (PARTITION BY l.CNPJ_FUNDO_CLASSE ORDER BY l.DT_COMPTC DESC) AS rn
    FROM lamina_lamina_fi l
),

-- Calculate recent performance metrics
recent_performance AS (
    SELECT
        fpi.fund_id.value AS fund_id,

        -- 2024 YTD performance
        AVG(CASE WHEN fpi.year = 2024 THEN fpi.return_pct END) AS return_ytd_2024_avg,
        SUM(CASE WHEN fpi.year = 2024 THEN fpi.return_pct END) AS return_ytd_2024_cumulative,

        -- Last 12 months
        AVG(CASE
            WHEN fpi.year = 2024 OR (fpi.year = 2023 AND fpi.month >= 11)
            THEN fpi.return_pct
        END) AS return_12m_avg,

        -- Last 6 months
        AVG(CASE
            WHEN fpi.year = 2024 AND fpi.month >= 5
            THEN fpi.return_pct
        END) AS return_6m_avg,

        -- Last 3 months
        AVG(CASE
            WHEN fpi.year = 2024 AND fpi.month >= 8
            THEN fpi.return_pct
        END) AS return_3m_avg,

        -- Last month performance
        (SELECT return_pct
         FROM fund_performance_indicators fpi2
         WHERE fpi2.fund_id.value = fpi.fund_id.value
         ORDER BY fpi2.year DESC, fpi2.month DESC
         LIMIT 1) AS return_last_month,

        -- Volatility (standard deviation of returns)
        STDDEV(CASE WHEN fpi.year >= 2023 THEN fpi.return_pct END) AS volatility_12m,

        -- Count negative months
        SUM(CASE WHEN fpi.return_pct < 0 AND fpi.year >= 2023 THEN 1 ELSE 0 END) AS negative_months_12m,

        -- Best and worst months
        MAX(CASE WHEN fpi.year >= 2023 THEN fpi.return_pct END) AS best_month_12m,
        MIN(CASE WHEN fpi.year >= 2023 THEN fpi.return_pct END) AS worst_month_12m

    FROM fund_performance_indicators fpi
    GROUP BY fpi.fund_id.value
)

-- Final view with all filterable fields
SELECT
    af.fund_id,
    af.cnpj,
    af.legal_name,
    lm.trade_name,

    -- Classification
    af.investment_class,
    af.anbima_classification,
    af.fund_type,
    af.structure,
    af.target_audience,
    lm.target_audience_desc,
    af.performance_benchmark,
    lm.benchmark_index,
    lm.risk_class,
    af.manager_type,
    af.status AS original_status,
    af.data_quality,

    -- Fund characteristics
    af.is_fund_of_funds,
    af.is_exclusive_fund,
    af.can_invest_abroad_100_pct,
    af.has_long_term_taxation,
    af.registration_date,
    af.constitution_date,
    af.activity_start_date,

    -- Size and fees
    COALESCE(lm.net_asset_value, af.nav_from_funds) AS nav,
    af.nav_currency,
    COALESCE(lm.management_fee_pct, af.management_fee_from_funds) AS management_fee_pct,
    COALESCE(lm.performance_fee_pct, af.performance_fee_from_funds) AS performance_fee_pct,
    lm.entry_fee_pct,
    lm.exit_fee_pct,
    lm.expense_ratio_pct,

    -- Investment constraints
    lm.min_initial_investment,
    lm.min_additional_investment,
    lm.min_redemption,
    lm.min_balance_required,
    lm.lockup_days,

    -- Liquidity
    lm.purchase_settlement_days,
    lm.redemption_settlement_days,
    lm.redemption_payment_days,

    -- Portfolio characteristics
    lm.foreign_assets_pct,
    lm.private_credit_pct,
    lm.leverage_ratio,

    -- Performance - Historical (5 year)
    lm.return_5y_pct,
    lm.benchmark_return_5y_pct,
    CASE
        WHEN lm.benchmark_return_5y_pct IS NOT NULL AND lm.benchmark_return_5y_pct != 0
        THEN lm.return_5y_pct - lm.benchmark_return_5y_pct
        ELSE NULL
    END AS excess_return_5y_pct,
    lm.years_with_losses,

    -- Performance - Recent
    rp.return_last_month,
    rp.return_3m_avg,
    rp.return_6m_avg,
    rp.return_12m_avg,
    rp.return_ytd_2024_avg,
    rp.return_ytd_2024_cumulative,

    -- Risk metrics
    rp.volatility_12m,
    rp.negative_months_12m,
    rp.best_month_12m,
    rp.worst_month_12m,
    CASE
        WHEN rp.volatility_12m IS NOT NULL AND rp.volatility_12m != 0
        THEN rp.return_12m_avg / rp.volatility_12m
        ELSE NULL
    END AS sharpe_ratio_approx,

    -- Metadata
    lm.lamina_report_date,
    af.timestamp AS last_updated

FROM active_funds af
LEFT JOIN lamina_metrics lm
    ON af.cnpj = lm.cnpj
    AND lm.rn = 1
LEFT JOIN recent_performance rp
    ON af.fund_id = rp.fund_id

ORDER BY af.legal_name;


-- ============================================================================
-- USAGE EXAMPLES FOR STRUCTURED FILTER VIEW
-- ============================================================================

-- Example 1: Find equity funds with >15% YTD returns in 2024
-- SELECT cnpj, legal_name, investment_class, return_ytd_2024_avg
-- FROM fund_structured_filter_view
-- WHERE investment_class = 'Ações'
--   AND return_ytd_2024_avg > 15.0
-- ORDER BY return_ytd_2024_avg DESC
-- LIMIT 10;

-- Example 2: Find low-fee multimercado funds with good performance
-- SELECT cnpj, legal_name, management_fee_pct, return_12m_avg, nav
-- FROM fund_structured_filter_view
-- WHERE investment_class = 'Multimercado'
--   AND management_fee_pct < 2.0
--   AND return_12m_avg > 10.0
--   AND nav > 10000000  -- At least R$10M AUM
-- ORDER BY return_12m_avg DESC;

-- Example 3: Find funds with low minimum investment (accessible to retail)
-- SELECT cnpj, legal_name, min_initial_investment, investment_class, return_ytd_2024_avg
-- FROM fund_structured_filter_view
-- WHERE min_initial_investment <= 1000
--   AND min_initial_investment IS NOT NULL
-- ORDER BY return_ytd_2024_avg DESC;

-- Example 4: Find funds that beat their benchmark by >5% over 5 years
-- SELECT cnpj, legal_name, return_5y_pct, benchmark_return_5y_pct, excess_return_5y_pct
-- FROM fund_structured_filter_view
-- WHERE excess_return_5y_pct > 5.0
-- ORDER BY excess_return_5y_pct DESC;

-- Example 5: Find low-volatility fixed income funds
-- SELECT cnpj, legal_name, volatility_12m, return_12m_avg, investment_class
-- FROM fund_structured_filter_view
-- WHERE investment_class = 'Renda Fixa'
--   AND volatility_12m < 0.5
--   AND volatility_12m IS NOT NULL
-- ORDER BY return_12m_avg DESC;

-- Example 6: Rank funds by Sharpe ratio (risk-adjusted returns)
-- SELECT cnpj, legal_name, return_12m_avg, volatility_12m, sharpe_ratio_approx
-- FROM fund_structured_filter_view
-- WHERE sharpe_ratio_approx IS NOT NULL
-- ORDER BY sharpe_ratio_approx DESC
-- LIMIT 20;
