-- Traceability metadata for ODB-like HDF5 ingest.
-- W1-W8 use dense benchmark IDs. W9-W11 additionally use these tables to
-- preserve/query source FE labels and Step/Frame meaning.

CREATE TABLE IF NOT EXISTS h5_node_source (
    ship_type          VARCHAR(50) NOT NULL,
    scale              VARCHAR(50) NOT NULL,
    zone_type          VARCHAR(50) NOT NULL,
    part_name          VARCHAR(160) NOT NULL,
    instance_name      VARCHAR(160) NOT NULL,
    node_id            INTEGER NOT NULL,
    source_node_label  BIGINT NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, node_id)
);

CREATE INDEX IF NOT EXISTS idx_h5_node_source_label
ON h5_node_source (ship_type, scale, zone_type, source_node_label);

CREATE TABLE IF NOT EXISTS h5_cell_source (
    ship_type             VARCHAR(50) NOT NULL,
    scale                 VARCHAR(50) NOT NULL,
    zone_type             VARCHAR(50) NOT NULL,
    part_name             VARCHAR(160) NOT NULL,
    instance_name         VARCHAR(160) NOT NULL,
    cell_id               INTEGER NOT NULL,
    source_element_label  BIGINT NOT NULL,
    element_type          VARCHAR(80) NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE TABLE IF NOT EXISTS h5_frame_metadata (
    ship_type          VARCHAR(50) NOT NULL,
    scale              VARCHAR(50) NOT NULL,
    zone_type          VARCHAR(50) NOT NULL,
    timestep           INTEGER NOT NULL,
    step_name          VARCHAR(160) NOT NULL,
    step_index         INTEGER NOT NULL,
    step_domain        INTEGER,
    frame_name         VARCHAR(160) NOT NULL,
    frame_index        INTEGER NOT NULL,
    inc_or_mode        INTEGER,
    time_or_frequency  DOUBLE PRECISION,
    description        TEXT NOT NULL DEFAULT '',
    load_case          VARCHAR(160) NOT NULL DEFAULT '',
    PRIMARY KEY (ship_type, scale, zone_type, timestep)
);

CREATE INDEX IF NOT EXISTS idx_h5_frame_source
ON h5_frame_metadata (ship_type, scale, zone_type, step_name, frame_index);
