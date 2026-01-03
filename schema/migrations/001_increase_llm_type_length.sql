-- Migration: Increase llm_type column length from VARCHAR(20) to VARCHAR(50)
-- Date: 2025-10-31
-- Reason: Support longer llm_type values like 'orchestrator_comprehensive' (27 chars)
--         which was failing with StringDataRightTruncation error
--
-- Current llm_type values include:
--   - 'sec', 'ta', 'full' (legacy)
--   - 'orchestrator_comprehensive', 'orchestrator_standard', 'orchestrator_quick'
--   - 'fundamental', 'technical', 'synthesis'
--
-- Usage:
--   PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database -f schema/migrations/001_increase_llm_type_length.sql

-- Check current column definition
SELECT
    table_name,
    column_name,
    data_type,
    character_maximum_length
FROM information_schema.columns
WHERE table_name = 'llm_responses' AND column_name = 'llm_type';

-- Alter the column to VARCHAR(50)
ALTER TABLE llm_responses
    ALTER COLUMN llm_type TYPE VARCHAR(50);

-- Verify the change
SELECT
    table_name,
    column_name,
    data_type,
    character_maximum_length
FROM information_schema.columns
WHERE table_name = 'llm_responses' AND column_name = 'llm_type';

-- Show affected rows (if any exist with long llm_type values)
SELECT COUNT(*) as total_llm_responses FROM llm_responses;
SELECT llm_type, COUNT(*) as count
FROM llm_responses
GROUP BY llm_type
ORDER BY llm_type;
