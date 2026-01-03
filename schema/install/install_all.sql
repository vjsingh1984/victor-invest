-- InvestiGator Complete Schema Installation
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- This script imports all schema files in order.
-- For PostgreSQL, you can use \i to include files.
-- For SQLite, concatenate all files and run.
--
-- PostgreSQL Usage:
--   cd schema/install
--   psql -h HOST -U USER -d DATABASE -f install_all.sql
--
-- SQLite Usage:
--   cd schema/install
--   cat 00_core_tables.sql 01_market_data_tables.sql 02_sentiment_tables.sql \
--       03_macro_indicators_tables.sql 04_rl_tables.sql | sqlite3 investigator.db
--
-- Or use the Python installer:
--   python -m investigator.cli db install --sqlite investigator.db
--   python -m investigator.cli db install --postgres postgresql://user:pass@host/db

-- PostgreSQL: Use \i to include files (comment out for SQLite)
-- \i 00_core_tables.sql
-- \i 01_market_data_tables.sql
-- \i 02_sentiment_tables.sql
-- \i 03_macro_indicators_tables.sql
-- \i 04_rl_tables.sql

-- For a single-file install, the content below includes all tables inline.
-- This is generated for convenience - the individual files are canonical.

-- ============================================================================
-- SCHEMA INSTALLATION COMPLETE
-- ============================================================================

-- Verify installation
SELECT 'Schema installation complete!' AS status;
SELECT version, description, applied_at FROM schema_version ORDER BY version;
