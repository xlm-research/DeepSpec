import config.distributed.profiles as profiles

globals().update(profiles.build(profiles.defaults(use_fsdp=False)))
