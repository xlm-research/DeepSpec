"""Transport matrix coverage is separate from feature-capacity acceptance."""

import pytest

from deepspec.pipeline.planning import build_plan
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize(
    "layout,writers,readers",
    [
        ("M0", {"node-a"}, {"node-a"}),
        ("M2-DP2", {"node-a"}, {"node-c"}),
        ("M3", {"node-a"}, {"node-b", "node-c"}),
    ],
)
def test_transport_covers_actual_writer_and_reader_nodes(layout, writers, readers):
    from deepspec.pipeline.transport import participants, validate_matrix

    config = task_config(layout)
    p = build_plan(
        config, node_facts(config), input_plan(config), run_id="probe-run", now=100
    ).to_dict()
    assert participants(p) == (writers, readers)
    matrix = [
        {
            "writer_node": w,
            "reader_node": r,
            "verified": True,
            "nbytes": 100,
            "expected_bytes": 100,
        }
        for w in writers
        for r in readers
    ]
    assert validate_matrix(p, matrix)
    with pytest.raises(ValueError, match="coverage"):
        validate_matrix(p, matrix[:-1])
    matrix[0]["verified"] = False
    with pytest.raises(ValueError, match="full"):
        validate_matrix(p, matrix)
    matrix[0]["verified"] = True
    matrix[0]["nbytes"] -= 1
    with pytest.raises(ValueError, match="full"):
        validate_matrix(p, matrix)
