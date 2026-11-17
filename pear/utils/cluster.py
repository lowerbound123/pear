from collections.abc import Mapping, Sequence


def check_single_node_split(config, nodes: Sequence[Mapping], available: Mapping) -> None:
    """Validate the small asynchronous run before creating Ray placement groups. No Ray side effects here."""
    trainer = config.trainer
    policy = config.actor_rollout_ref
    rollout = policy.rollout
    fsdp = policy.actor.fsdp_config
    sizes = (
        trainer.nnodes, trainer.n_gpus_per_node, rollout.nnodes, rollout.n_gpus_per_node,
        rollout.tensor_model_parallel_size, fsdp.fsdp_size, fsdp.ulysses_sequence_parallel_size,
    )
    if sizes != (1,) * len(sizes) or policy.hybrid_engine:
        raise ValueError("single-node split requires one trainer GPU and one rollout GPU, with TP/SP/FSDP size 1")

    trainer_label = trainer.node_label
    rollout_label = rollout.custom.get("node_label")
    for node in nodes:
        resources = node.get("Resources", {})
        if node.get("Alive") and all(
            resources.get(name, 0) >= count
            for name, count in (("GPU", 2), (trainer_label, 1), (rollout_label, 1), ("libero", 1))
        ):
            if available.get("GPU", 0) >= 2:
                return
            raise ValueError("single-node split needs two free GPUs; stop the other GPU workload before launching")
    raise ValueError(
        "single-node split needs a live Ray node with two GPUs and trainer, rollout and libero resources; "
        "start scripts/start_ray_node.sh combined"
    )
