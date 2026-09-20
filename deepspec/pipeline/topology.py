"""Producer routing and Titan's DP-major, TP-minor reader ownership."""


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def expected_counts(config, *, sample_count):
    training = config["training"]
    steps = _positive_integer(training["steps"], "training.steps")
    batch = _positive_integer(training["global_batch_size"], "training.global_batch_size")
    dp = _positive_integer(training["dp"], "training.dp")
    tp = _positive_integer(training["tp"], "training.tp")
    if batch != 4 or tp != 4 or dp not in (1, 2) or batch % dp:
        raise ValueError("Training requires TP4, DP1/2 and complete global batches of four")
    samples = steps * batch
    if type(sample_count) is not int or sample_count != samples:
        raise ValueError(f"Input plan must contain exactly {samples} samples in complete updates")
    return {"samples": samples, "optimizer_steps": steps, "gas": batch // dp,
            "native_cursor": samples // dp, "sample_cursor": samples, "reader_count": samples * tp}


def planned_producer(config, position):
    if type(position) is not int or position < 0:
        raise ValueError("Sample position must be a nonnegative integer")
    return position % _positive_integer(config["inference"]["dp"], "inference.dp")


def planned_readers(config, position):
    if type(position) is not int or position < 0:
        raise ValueError("Sample position must be a nonnegative integer")
    dp = _positive_integer(config["training"]["dp"], "training.dp")
    tp = _positive_integer(config["training"]["tp"], "training.tp")
    start = (position % dp) * tp
    return tuple(range(start, start + tp))


def training_rank_table(nodes, *, tp, dp):
    """Use node-local visible slots; physical UUIDs are bound at allocation."""
    from .planning import TrainingParticipant

    _positive_integer(tp, "training.tp")
    _positive_integer(dp, "training.dp")
    sizes = [_positive_integer(size, "local_world_size") for _, size in nodes]
    if not sizes or len(set(sizes)) != 1 or sum(sizes) != tp * dp or any(n % tp for n in sizes):
        raise ValueError("Training nodes must have uniform complete TP groups and TP*DP ranks")
    if len({node for node, _ in nodes}) != len(nodes):
        raise ValueError("Training nodes must be unique")
    result = []
    for node_rank, (node_id, local_world) in enumerate(nodes):
        for local_rank in range(local_world):
            rank = len(result)
            result.append(TrainingParticipant(rank, node_rank, local_rank, local_world, node_id,
                                              rank % tp, rank // tp, local_rank).to_dict())
    return result


def producer_dp(config):
    dp = config.get("producer_dp", 1)
    if type(dp) is not int or dp < 1 or (dp not in (1, 2) and config.get("approved_inference_dp") != dp):
        raise ValueError("Producer DP must be positive and approved by the topology plan")
    return dp


def sample_producer(config, position):
    return position % producer_dp(config)


def consumer_dp(config):
    dp = config.get("consumer_dp", 1)
    world = config["consumer_world_size"]
    if dp < 1 or world % dp:
        raise ValueError("Consumer world size must contain complete DP/TP groups")
    return dp


def consumer_nodes(config):
    # Old saved runs used one DP group per node. New launches separate roles.
    nodes = config.get("consumer_nodes", consumer_dp(config))
    if nodes < 1 or config["consumer_world_size"] % nodes:
        raise ValueError("Consumer ranks must divide evenly across launcher nodes")
    return nodes


def sample_readers(config, position):
    dp = consumer_dp(config)
    tp = config["consumer_world_size"] // dp
    start = (position % dp) * tp
    return tuple(range(start, start + tp))


def consumer_microbatches(config):
    """Native checkpoint cursor counts synchronized DP microsteps, not samples."""
    dp = consumer_dp(config)
    count, remainder = divmod(len(config["samples"]), dp)
    if remainder:
        raise ValueError("Input plan ends inside a DP microstep")
    return count
