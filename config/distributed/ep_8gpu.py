"""Validation profile: the current dense DFlash2 model rejects EP > 1."""

import config.distributed.profiles as profiles

globals().update(profiles.build(profiles.defaults(dp_shard=8, ep=8)))
