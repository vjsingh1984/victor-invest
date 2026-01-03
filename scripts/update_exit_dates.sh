#!/bin/bash
# Periodic update of exit_dates for records missing them
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database -c "
UPDATE valuation_outcomes
SET per_model_rewards = jsonb_set(
    per_model_rewards,
    '{multi_period,exit_dates}',
    jsonb_build_object(
        '1m', (COALESCE(entry_date, analysis_date) + INTERVAL '30 days')::date::text,
        '3m', (COALESCE(entry_date, analysis_date) + INTERVAL '90 days')::date::text,
        '6m', (COALESCE(entry_date, analysis_date) + INTERVAL '180 days')::date::text,
        '12m', (COALESCE(entry_date, analysis_date) + INTERVAL '365 days')::date::text,
        '18m', (COALESCE(entry_date, analysis_date) + INTERVAL '540 days')::date::text,
        '24m', (COALESCE(entry_date, analysis_date) + INTERVAL '730 days')::date::text,
        '36m', (COALESCE(entry_date, analysis_date) + INTERVAL '1095 days')::date::text
    ),
    true
)
WHERE per_model_rewards->'multi_period' IS NOT NULL
  AND (per_model_rewards->'multi_period'->'exit_dates' IS NULL 
       OR per_model_rewards->'multi_period'->'exit_dates'->>'12m' IS NULL);
"
