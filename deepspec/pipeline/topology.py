"""Producer routing and Titan's DP-major, TP-minor reader ownership."""


def producer_dp(config):
    dp = config.get("producer_dp", 1)
    if dp not in (1, 2):
        raise ValueError("Producer DP must be one or two")
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
