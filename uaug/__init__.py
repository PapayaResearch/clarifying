from omegaconf import OmegaConf

# Arithmetic resolver
OmegaConf.register_new_resolver("add", lambda a, b: a + b, replace=True)
