-- ============================================================
-- 网格几何与拓扑表（无 timestep）
-- 用于高效邻接查询与射线-网格相交
-- ============================================================

-- 1. 元数据（可选）
CREATE TABLE IF NOT EXISTS mesh_metadata (
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    node_count   INTEGER NOT NULL,
    element_count INTEGER NOT NULL,
    face_count   INTEGER NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type)
);

-- 2. 单元邻接（一单元一行，邻居为整数数组）
CREATE TABLE IF NOT EXISTS cell_adjacency (
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    cell_id      INTEGER NOT NULL,
    neighbor_ids INTEGER[] NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_adjacency_lookup
ON cell_adjacency (ship_type, scale, zone_type, cell_id);

CREATE INDEX IF NOT EXISTS idx_cell_adjacency_gin
ON cell_adjacency USING GIN (neighbor_ids);

-- 3. 单元中心坐标
CREATE TABLE IF NOT EXISTS cell_centroid (
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    cell_id      INTEGER NOT NULL,
    x            DOUBLE PRECISION NOT NULL,
    y            DOUBLE PRECISION NOT NULL,
    z            DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_centroid_lookup
ON cell_centroid (ship_type, scale, zone_type, cell_id);

CREATE INDEX IF NOT EXISTS idx_cell_centroid_xyz
ON cell_centroid (ship_type, scale, zone_type, x, y, z);

-- 4. 共享面平面方程（用于精确射线-面求交）
-- 平面: nx*x + ny*y + nz*z + d = 0（法向建议单位化）
CREATE TABLE IF NOT EXISTS cell_face_plane (
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    cell_id      INTEGER NOT NULL,
    neighbor_id  INTEGER NOT NULL,
    nx           DOUBLE PRECISION NOT NULL,
    ny           DOUBLE PRECISION NOT NULL,
    nz           DOUBLE PRECISION NOT NULL,
    d            DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id, neighbor_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_face_plane_lookup
ON cell_face_plane (ship_type, scale, zone_type, cell_id);

-- 5. 单元节点列表（用于平面-单元相交等需要按节点判断的几何运算）
CREATE TABLE IF NOT EXISTS cell_nodes (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    cell_id     INTEGER NOT NULL,
    node_ids    INTEGER[] NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_nodes_lookup
ON cell_nodes (ship_type, scale, zone_type, cell_id);

-- 6. 节点坐标（可选；节点量大时可用 zone_nodes 单行+数组替代）
CREATE TABLE IF NOT EXISTS node_coordinates (
    ship_type    VARCHAR(50) NOT NULL,
    scale        VARCHAR(50) NOT NULL,
    zone_type    VARCHAR(50) NOT NULL,
    node_id      INTEGER NOT NULL,
    x            DOUBLE PRECISION NOT NULL,
    y            DOUBLE PRECISION NOT NULL,
    z            DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, node_id)
);

CREATE INDEX IF NOT EXISTS idx_node_coordinates_lookup
ON node_coordinates (ship_type, scale, zone_type, node_id);

-- ============================================================
-- 标量场（用于等值面 iso-surface）
-- 说明：原始 .dat 中 U/V/W/P/K/E 是 cell-centered（每个 cell 一个值）。
--      等值面提取（Marching Cubes/Tetra）更适合用 node-centered 标量，
--      因此提供从 cell_scalar → node_scalar 的预计算表。
-- ============================================================

-- 7. cell-centered 标量（来自 .dat 的 Element_Variables）
CREATE TABLE IF NOT EXISTS cell_scalar (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    timestep    INTEGER NOT NULL,
    var         VARCHAR(10) NOT NULL,   -- 'U','V','W','P','K','E'
    cell_id     INTEGER NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, timestep, var, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_cell_scalar_lookup
ON cell_scalar (ship_type, scale, zone_type, timestep, var, cell_id);

-- 8. node-centered 标量（由 cell_scalar 插值得到，用于等值面）
CREATE TABLE IF NOT EXISTS node_scalar (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    timestep    INTEGER NOT NULL,
    var         VARCHAR(10) NOT NULL,   -- 'U','V','W','P','K','E'
    node_id     INTEGER NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, timestep, var, node_id)
);

CREATE INDEX IF NOT EXISTS idx_node_scalar_lookup
ON node_scalar (ship_type, scale, zone_type, timestep, var, node_id);

-- 9. node↔cell 关系（便于从 cell_scalar 聚合/插值得 node_scalar）
CREATE TABLE IF NOT EXISTS node_cells (
    ship_type   VARCHAR(50) NOT NULL,
    scale       VARCHAR(50) NOT NULL,
    zone_type   VARCHAR(50) NOT NULL,
    node_id     INTEGER NOT NULL,
    cell_id     INTEGER NOT NULL,
    PRIMARY KEY (ship_type, scale, zone_type, node_id, cell_id)
);

CREATE INDEX IF NOT EXISTS idx_node_cells_node
ON node_cells (ship_type, scale, zone_type, node_id);

-- ============================================================
-- 常用查询示例（占位符 :ship_type, :scale, :zone_type, :k 由应用替换）
-- ============================================================

-- 查询单元 k 的邻居及其中心坐标
-- SELECT c.cell_id, c.x, c.y, c.z
-- FROM cell_centroid c
-- JOIN (
--     SELECT unnest(a.neighbor_ids) AS adj_cell_id
--     FROM cell_adjacency a
--     WHERE a.ship_type = $1 AND a.scale = $2 AND a.zone_type = $3 AND a.cell_id = $4
-- ) adj ON c.ship_type = $1 AND c.scale = $2 AND c.zone_type = $3 AND c.cell_id = adj.adj_cell_id;

-- 射线步进：取当前单元所有面的平面与邻居
-- SELECT neighbor_id, nx, ny, nz, d
-- FROM cell_face_plane
-- WHERE ship_type = $1 AND scale = $2 AND zone_type = $3 AND cell_id = $4;
