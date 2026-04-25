-- retention.sql: Periodic cleanup (run via pg_cron or external cron).
-- Recommended schedule: daily at 03:00 UTC.

DELETE FROM equity_snapshots WHERE timestamp < now() - INTERVAL '90 days';
DELETE FROM health_snapshots WHERE timestamp < now() - INTERVAL '30 days';
VACUUM ANALYZE equity_snapshots;
VACUUM ANALYZE health_snapshots;
