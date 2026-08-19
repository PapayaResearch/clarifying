from omegaconf import OmegaConf

# Arithmetic resolvers
OmegaConf.register_new_resolver("add", lambda a, b: a + b, replace=True)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
