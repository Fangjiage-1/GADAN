"""Safe checkpoint restoration for GADAN evaluation and resume."""


ARCHITECTURE_ARGUMENTS = (
    "text_encoder_type",
    "tokenizer_path",
    "text_encoder_path",
    "bilstm_embed_dim",
    "bilstm_hidden_dim",
    "bilstm_num_layers",
    "bilstm_dropout",
    "gsbi_d_geo",
    "backbone",
    "dilation",
    "position_embedding",
    "hidden_dim",
    "nheads",
    "enc_layers",
    "dec_layers",
    "dim_feedforward",
    "dropout",
    "num_queries",
    "num_feature_levels",
    "enc_n_points",
    "dec_n_points",
    "with_box_refine",
    "two_stage",
    "binary",
    "num_frames",
)


def _checkpoint_arg(checkpoint_args, name):
    if isinstance(checkpoint_args, dict):
        if name not in checkpoint_args:
            raise KeyError(name)
        return checkpoint_args[name]
    if not hasattr(checkpoint_args, name):
        raise KeyError(name)
    return getattr(checkpoint_args, name)


def restore_architecture_args(runtime_args, checkpoint):
    """Restore every architecture-defining option before constructing a model."""
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None:
        raise RuntimeError(
            "Checkpoint has no saved args; refusing to infer the model architecture."
        )

    missing = []
    restored = {}
    for name in ARCHITECTURE_ARGUMENTS:
        try:
            value = _checkpoint_arg(checkpoint_args, name)
        except KeyError:
            missing.append(name)
            continue
        setattr(runtime_args, name, value)
        restored[name] = value

    if missing:
        raise RuntimeError(
            "Checkpoint is missing architecture metadata: " + ", ".join(missing)
        )

    print("Restored checkpoint architecture:")
    for name, value in restored.items():
        print(f"  {name}={value}")


def load_model_state_strict(model, checkpoint):
    """Load model weights and fail on any missing or unexpected parameter."""
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid 'model' state dict.")

    # Profiling tools may append these non-model counters.
    state_dict = {
        name: value
        for name, value in state_dict.items()
        if not (name.endswith("total_params") or name.endswith("total_ops"))
    }
    model.load_state_dict(state_dict, strict=True)
