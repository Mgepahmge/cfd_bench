-- Spatial modeling tables for Workload4/5 acceleration.
-- Independent from existing create_mesh_tables.sql.

CREATE EXTENSION IF NOT EXISTS postgis;

-- 1) Full-cell geometry layer for spatial candidate filtering.
CREATE TABLE IF NOT EXISTS cell_geom_full (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    cell_id     INTEGER NOT NULL,
    centroid    geometry(PointZ, 0) NOT NULL,
    geom        geometry(MultiPolygonZ, 0) NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_geom_full_lookup
ON cell_geom_full (ship_type, scale, zone_type, cell_id);

CREATE INDEX IF NOT EXISTS idx_cell_geom_full_geom_gist
ON cell_geom_full USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_cell_geom_full_centroid_gist
ON cell_geom_full USING GIST (centroid);

-- 2) Point-location acceleration buckets (regular grid over centroids).
-- Note: bbox_* are numeric 3D ranges; bbox_xy is optional 2D geometry helper.
CREATE TABLE IF NOT EXISTS point_locator_grid (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    bucket_id   BIGINT NOT NULL,
    ix          INTEGER NOT NULL,
    iy          INTEGER NOT NULL,
    iz          INTEGER NOT NULL,
    x_min       DOUBLE PRECISION NOT NULL,
    x_max       DOUBLE PRECISION NOT NULL,
    y_min       DOUBLE PRECISION NOT NULL,
    y_max       DOUBLE PRECISION NOT NULL,
    z_min       DOUBLE PRECISION NOT NULL,
    z_max       DOUBLE PRECISION NOT NULL,
    bbox_xy     geometry(Polygon, 0) NOT NULL,
    cell_ids    INTEGER[] NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, bucket_id)
);

CREATE INDEX IF NOT EXISTS idx_point_locator_grid_lookup
ON point_locator_grid (ship_type, scale, zone_type, ix, iy, iz);

CREATE INDEX IF NOT EXISTS idx_point_locator_grid_bbox_xy_gist
ON point_locator_grid USING GIST (bbox_xy);

CREATE INDEX IF NOT EXISTS idx_point_locator_grid_cells_gin
ON point_locator_grid USING GIN (cell_ids);

