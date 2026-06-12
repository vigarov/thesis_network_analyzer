# Keys omitted from the training fingerprint and from training-config equality checks.
# `framework_version` is ignored if present in an old config.json on disk.
FINGERPRINT_EXCLUDE_KEYS = frozenset(
	{
		"internal_keep_tensors",
		"device",
		"save_model_cp",
		"save_model",
		"force",
		"training_config_fingerprint",
		"framework_version",
	}
)

# placeholder expanded to all available optimizers/experiments (depending on context)
ALL_AVAILABLE_SELECTION = "all"

# Human-friendly optimizer CLI names -> registry extractor class name.
# Kept in sync with `OPTIMIZER_SHORTHAND` in `run_simulation` (derived from this dict).
INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS: dict[str, str] = {
	"sgd": "SGDExtractor",
	"adam": "AdamExtractor",
	"adagrad": "AdaGradExtractor",
	"pure_shampoo": "PureShampooExtractor",
	"grafted_shampoo": "GraftedShampooExtractor",
}

# Checkpoint filename inside each `use_initial_model` bundle subfolder (try first match).
INITIAL_MODEL_CHECKPOINT_FILENAMES: tuple[str, ...] = ("model.pt", "model.pth")

# Required optimizer state filename inside each `use_initial_model` bundle subfolder.
INITIAL_MODEL_OPTIMIZER_STATE_FILENAME = "optimizer.pt"
