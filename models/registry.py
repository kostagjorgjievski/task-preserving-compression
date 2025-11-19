from .cnn_ae import CNNAutoencoder


MODEL_REGISTRY = {
    "cnn_ae": CNNAutoencoder,
    # "tcn_ae": TCNAutoencoder,
    # "transformer_ae": TransformerAutoencoder,
}


def build_model(cfg, num_channels):
    mcfg = cfg["model"]
    model_type = mcfg["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(
        input_channels=num_channels,
        latent_dim=mcfg["latent_dim"],
        input_length=mcfg["input_length"],
    )
    return model
