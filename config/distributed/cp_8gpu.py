import config.distributed.profiles as profiles

globals().update(
    profiles.build(
        profiles.defaults(cp=8, context_parallel_backend="model_native")
    )
)
