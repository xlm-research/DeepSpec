import config.distributed.profiles as profiles

globals().update(profiles.build(profiles.defaults(dp_shard=2, tp=4)))
