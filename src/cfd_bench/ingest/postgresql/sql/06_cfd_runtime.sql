-- Legacy CFD-only runtime acceleration.  H5/structural datasets do not need
-- this table; adding it is harmless and leaves their existing storage layout
-- untouched.
CREATE TABLE IF NOT EXISTS cell_bounds (
    ship_type VARCHAR(50) NOT NULL,
    scale VARCHAR(50) NOT NULL,
    zone_type VARCHAR(50) NOT NULL,
    cell_id INTEGER NOT NULL,
    xmin DOUBLE PRECISION NOT NULL,
    xmax DOUBLE PRECISION NOT NULL,
    ymin DOUBLE PRECISION NOT NULL,
    ymax DOUBLE PRECISION NOT NULL,
    zmin DOUBLE PRECISION NOT NULL,
    zmax DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_bounds_dataset
ON cell_bounds (ship_type, scale, zone_type, cell_id);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_x
ON cell_bounds (ship_type, scale, zone_type, xmin, xmax);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_y
ON cell_bounds (ship_type, scale, zone_type, ymin, ymax);
CREATE INDEX IF NOT EXISTS idx_cell_bounds_z
ON cell_bounds (ship_type, scale, zone_type, zmin, zmax);
