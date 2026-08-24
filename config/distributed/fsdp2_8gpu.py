import config.distributed.profiles as profiles

globals().update(
    profiles.build(
        profiles.defaults(dp_shard=8, use_activation_checkpoint=True)
    )
)
