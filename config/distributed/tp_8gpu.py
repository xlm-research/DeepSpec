import config.distributed.profiles as profiles

globals().update(profiles.build(profiles.defaults(tp=8, use_fsdp=False)))
