-- Boundary face model for Workload6 resistance analysis.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS boundary_face_geom (
    face_id      BIGSERIAL PRIMARY KEY,
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    cell_id      INTEGER NOT NULL,
    area         DOUBLE PRECISION NOT NULL,
    nx           DOUBLE PRECISION NOT NULL,
    ny           DOUBLE PRECISION NOT NULL,
    nz           DOUBLE PRECISION NOT NULL,
    geom         geometry(PolygonZ, 0) NOT NULL,
    patch_name   VARCHAR(80) NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_boundary_face_lookup
ON boundary_face_geom (ship_type, scale, zone_type, cell_id, patch_name);

CREATE INDEX IF NOT EXISTS idx_boundary_face_geom_gist
ON boundary_face_geom USING GIST (geom);

