"""MoE profile template: the current dense DFlash2 model rejects EP > 1."""

import config.distributed.profiles as profiles

globals().update(profiles.build(profiles.defaults(dp_shard=2, tp=4, ep=4)))
