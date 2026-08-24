import config.distributed.profiles as profiles

globals().update(
    profiles.build(
        profiles.defaults(
            dp_shard=2,
            tp=2,
            cp=2,
            context_parallel_backend="model_native",
        )
    )
)
