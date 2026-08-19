-- Legacy CFD-only runtime acceleration.  H5/structural datasets do not need
-- this table; adding it is harmless and leaves their existing storage layout
-- untouched.  Do not use PostgreSQL system-column names (xmin/xmax) here.
CREATE TABLE IF NOT EXISTS cell_bounds (
    ship_type VARCHAR(50) NOT NULL,
    scale VARCHAR(50) NOT NULL,
    zone_type VARCHAR(50) NOT NULL,
    cell_id INTEGER NOT NULL,
    min_x DOUBLE PRECISION NOT NULL,
    max_x DOUBLE PRECISION NOT NULL,
    min_y DOUBLE PRECISION NOT NULL,
    max_y DOUBLE PRECISION NOT NULL,
    min_z DOUBLE PRECISION NOT NULL,
    max_z DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_bounds_dataset
ON cell_bounds (ship_type, scale, zone_type, cell_id);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_x
ON cell_bounds (ship_type, scale, zone_type, min_x, max_x);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_y
ON cell_bounds (ship_type, scale, zone_type, min_y, max_y);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_z
ON cell_bounds (ship_type, scale, zone_type, min_z, max_z);
