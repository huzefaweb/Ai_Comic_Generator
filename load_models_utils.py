import yaml
import torch
import os
from diffusers import StableDiffusionXLPipeline
from utils import PhotoMakerStableDiffusionXLPipeline

def get_models_dict(config_path='config/models.yaml', verbose=False):
    """
    Loads model configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        verbose (bool): If True, prints the loaded configuration.

    Returns:
        dict: Parsed YAML data.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            if verbose:
                print("Loaded model configuration:", data)
            return data
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML file: {exc}")

def load_models(model_info, device="cuda", photomaker_path=None):
    """
    Loads a Stable Diffusion XL model or a PhotoMaker variant based on the provided info.

    Args:
        model_info (dict): Model configuration dictionary.
        device (str): Target device ('cuda' or 'cpu').
        photomaker_path (str, optional): Path to PhotoMaker adapter weights if using Photomaker.

    Returns:
        DiffusionPipeline: Loaded diffusion pipeline.
    """
    path = model_info.get("path")
    single_file = model_info.get("single_files", False)
    use_safetensors = model_info.get("use_safetensors", True)
    model_type = model_info.get("model_type", "original")

    if not path:
        raise ValueError("Model path must be specified in the model_info.")

    if model_type == "original":
        pipeline_cls = StableDiffusionXLPipeline
    elif model_type == "Photomaker":
        pipeline_cls = PhotoMakerStableDiffusionXLPipeline
    else:
        raise NotImplementedError(
            f"Unsupported model type '{model_type}'. Choose either 'original' or 'Photomaker'."
        )

    # Load model
    if single_file:
        print(f"Loading model from a single file: {path}")
        pipe = pipeline_cls.from_single_file(path, torch_dtype=torch.float16)
    else:
        print(f"Loading model from a directory: {path}")
        pipe = pipeline_cls.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=use_safetensors)

    pipe = pipe.to(device)

    # Load PhotoMaker adapter if needed
    if model_type == "Photomaker":
        if not photomaker_path:
            raise ValueError("Photomaker model type requires a valid 'photomaker_path'.")
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"
        )
        pipe.fuse_lora()

    return pipe