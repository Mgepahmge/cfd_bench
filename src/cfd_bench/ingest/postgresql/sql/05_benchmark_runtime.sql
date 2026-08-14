-- Runtime metadata derived during ingest so workloads do not depend on
-- machine-specific sidecar directories.

CREATE TABLE IF NOT EXISTS benchmark_max_diff (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    timestep    INTEGER NOT NULL,
    var         VARCHAR(64) NOT NULL,
    max_diff    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, timestep, var)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_max_diff_lookup
ON benchmark_max_diff (ship_type, scale, zone_type, timestep, var);
