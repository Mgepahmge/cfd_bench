from __future__ import annotations

import numpy as np
import pytest

from cfd_bench.features.structure_cfd_coupling.alignment import estimate_similarity_alignment


def test_similarity_alignment_recovers_known_uniform_transform_without_correspondence_ids():
    rng = np.random.default_rng(7)
    x = rng.uniform(-5.0, 5.0, 3000)
    half = 1.0 - (np.abs(x) / 5.0) ** 1.4
    source = np.column_stack(
        [
            x,
            half * rng.uniform(-1.0, 1.0, x.size),
            half * rng.uniform(-0.4, 0.8, x.size) + 0.03 * x,
        ]
    )
    angle = np.deg2rad(13.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 0.001
    translation = np.asarray([1.3, -0.25, 0.6], dtype=np.float64)
    target = scale * (source @ rotation.T) + translation

    result = estimate_similarity_alignment(
        source,
        target,
        max_points=3000,
        max_iterations=20,
        trim_fraction=0.8,
    )

    assert result.transform.scale == pytest.approx(scale, rel=1.0e-7)
    np.testing.assert_allclose(result.transform.rotation, rotation, atol=1.0e-7)
    np.testing.assert_allclose(result.transform.translation, translation, atol=1.0e-7)
    assert result.rmse_after < 1.0e-8
    assert result.confidence == "high"


def test_auto_alignment_requires_hull_reference_unless_explicitly_overridden():
    from cfd_bench.features.structure_cfd_coupling.engine import StructureCfdCouplingEngine

    with pytest.raises(ValueError, match="hull/reference surface zone"):
        StructureCfdCouplingEngine._resolve_alignment_zone(
            {"zones": ("0_Fluid", "0_InVel_top")}, "0_Fluid", None
        )
    assert (
        StructureCfdCouplingEngine._resolve_alignment_zone(
            {"zones": ("0_Fluid",)}, "0_Fluid", "custom_surface"
        )
        == "custom_surface"
    )
