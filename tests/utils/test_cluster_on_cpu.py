from types import SimpleNamespace

import pytest

from pear.utils.cluster import check_single_node_split


def config():
    return SimpleNamespace(
        trainer=SimpleNamespace(nnodes=1, n_gpus_per_node=1, node_label="trainer"),
        actor_rollout_ref=SimpleNamespace(
            hybrid_engine=False,
            actor=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_size=1, ulysses_sequence_parallel_size=1)),
            rollout=SimpleNamespace(
                nnodes=1, n_gpus_per_node=1, tensor_model_parallel_size=1, custom={"node_label": "rollout"}
            ),
        ),
    )


def test_both_roles_share_one_node_but_reserve_two_gpus():
    node = {"Alive": True, "Resources": {"GPU": 2, "trainer": 1, "rollout": 1, "libero": 4}}
    check_single_node_split(config(), [node], {"GPU": 2})


@pytest.mark.parametrize(
    "nodes, available",
    [
        ([{"Alive": True, "Resources": {"GPU": 1, "trainer": 1, "rollout": 1, "libero": 4}}], {"GPU": 1}),
        ([{"Alive": False, "Resources": {"GPU": 2, "trainer": 1, "rollout": 1, "libero": 4}}], {"GPU": 2}),
        ([{"Alive": True, "Resources": {"GPU": 2, "trainer": 1, "rollout": 1}}], {"GPU": 2}),
        ([{"Alive": True, "Resources": {"GPU": 2, "trainer": 1, "rollout": 1, "libero": 4}}], {"GPU": 1}),
        ([{"Alive": True, "Resources": {"GPU": 1, role: 1, "libero": 4}} for role in ("trainer", "rollout")], {"GPU": 2}),
    ],
)
def test_missing_busy_or_separate_resources_fail_before_placement(nodes, available):
    with pytest.raises(ValueError, match="single-node split"):
        check_single_node_split(config(), nodes, available)


def test_an_override_cannot_accidentally_request_four_gpus():
    settings = config()
    settings.trainer.n_gpus_per_node = 2
    with pytest.raises(ValueError, match="one trainer GPU and one rollout GPU"):
        check_single_node_split(settings, [], {})
