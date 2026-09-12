# Samples per training trial (checkpoint spacing for activation windows).
N_SAMPLES_PER_TRIAL: int = 100

# When True, saliency similarity uses complex-wavelet SSIM (steerable pyramid path).
USE_CW_SSIM: bool = True

# Steerable pyramid depth/orientations for CW-SSIM (must match `init_cw_ssim_pyramid` usage).
PYRAMID_LEVELS: int = 3
WAVELET_ORIENTATIONS: int = 8

# Spatial shape of MNIST inputs / saliency maps (H, W).
MNIST_IMAGE_SHAPE = (28, 28)

# Maps optimizer id prefix (before `_lr`) to registry class name for expert extraction.
OPTIMIZER_PREFIX_TO_REGISTRY_CLASS = {
    "sgd": "SGDExtractor",
    "adam": "AdamExtractor",
    "adagrad": "AdaGradExtractor",
    "pure_shampoo": "PureShampooExtractor",
    "grafted_shampoo": "GraftedShampooExtractor",
    "graft_shampoo": "GraftedShampooExtractor",
}
