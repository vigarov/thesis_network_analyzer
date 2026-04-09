# dict/json keys for optimizer_config.json
OPTIMIZER_TYPE_KEY = "type"
OPTIMIZER_LR_KEY = "lr"
OPTIMIZER_BETAS_KEY = "betas"
OPTIMIZER_EPS_KEY = "eps"
OPTIMIZER_MOMENTUM_KEY = "momentum"
SHAMPOO_PRECONDITIONER_EPSILON_KEY = "shampoo_preconditioner_epsilon"
GRAFTING_EPS_KEY = "grafting_eps"

# Default hyperparameters
DEFAULT_LR = 1e-3
DEFAULT_ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
DEFAULT_ADAM_EPS = 1e-8
DEFAULT_ADAGRAD_EPS = 1e-10
DEFAULT_SGD_MOMENTUM = 0.0
DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON = 1e-12
DEFAULT_SHAMPOO_BETAS: tuple[float, float] = (0.9, 0.999) # Coefficients used for computing running averages of gradient and its square
DEFAULT_GRAFTING_EPS = 1e-10
