"""Thin structured wrapper around the existing IoTDB interpolation engine."""

from __future__ import annotations

from typing import List

import numpy as np

from cfd_bench.core.warnings_state import preserve_warning_filters

from .schemas import (
    InterpolationPointResult,
    InterpolationRequest,
    InterpolationResponse,
)


def _result_passed(result) -> bool:
    # Keep validation identical to cfd_bench.cli.interpolate_cmd._result_passed
    # without parsing CLI stdout.
    return bool(
        np.isfinite(result.reconstruction_error)
        and result.reconstruction_error
        <= 1.0e-7 * max(1.0, float(np.linalg.norm(result.target)))
        and np.all(np.isfinite(result.weights))
        and abs(float(np.sum(result.weights)) - 1.0) <= 1.0e-8
        and all(np.isfinite(float(v)) for v in result.values.values())
    )


def interpolate_points(request: InterpolationRequest) -> InterpolationResponse:
    # Mirror the CLI's warning-filter isolation and reuse one IoTDB session for
    # the entire batch of points.
    with preserve_warning_filters():
        from cfd_bench.features.fluid_interpolation import FluidInterpolationEngine
        from cfd_bench.infra.iotdb.config import IoTDBConfig
        from cfd_bench.infra.iotdb.repository import IoTDBRepository

        repo = IoTDBRepository(IoTDBConfig())
        repo.open()
        try:
            engine = FluidInterpolationEngine(repo)
            items: List[InterpolationPointResult] = []
            resolved_zone = request.zone
            for point in request.points:
                result = engine.interpolate(
                    request.dataset,
                    int(request.step),
                    point,
                    variables=request.variables,
                    zone=request.zone,
                )
                resolved_zone = result.zone
                payload = {
                    "point": [float(x) for x in result.target.tolist()],
                    "values": {str(k): float(v) for k, v in result.values.items()},
                    "validation": "PASS" if _result_passed(result) else "FAIL",
                }
                if request.diagnostics:
                    payload.update(
                        {
                            "cell_id": int(result.cell_id),
                            "source_element_id": int(result.source_element_id),
                            "cell_node_ids": [int(x) for x in result.cell_node_ids],
                            "support_node_ids": [int(x) for x in result.support_node_ids],
                            "support_source_node_ids": [
                                int(x) for x in result.support_source_node_ids
                            ],
                            "weights": [float(x) for x in result.weights.tolist()],
                            "reconstruction_error": float(result.reconstruction_error),
                            "vertex_value_source": str(result.vertex_value_source),
                            "support_vertex_values": {
                                str(k): [float(x) for x in values]
                                for k, values in result.support_vertex_values.items()
                            },
                        }
                    )
                items.append(InterpolationPointResult(**payload))
        finally:
            repo.close()

    return InterpolationResponse(
        dataset=request.dataset,
        step=int(request.step),
        zone=str(resolved_zone or "0_Fluid"),
        results=items,
    )
