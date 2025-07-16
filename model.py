# model.py
# Merge image encoder and fuse module to create an ID Encoder
# Allows multiple ID images to update the text encoder with a stacked ID embedding.

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig

# Vision backbone configuration for the CLIP-based encoder
VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class MLP(nn.Module):
    """Simple MLP block with optional residual connection."""
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim, "Input and output dimensions must match when using residual."
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x += residual
        return x

class FuseModule(nn.Module):
    """Module that fuses prompt embeddings with ID embeddings."""
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        """Performs two-step fusion of prompt and ID embeddings."""
        stacked = torch.cat([prompt_embeds, id_embeds], dim=-1)
        fused = self.mlp1(stacked) + prompt_embeds
        fused = self.mlp2(fused)
        return self.layer_norm(fused)

    def forward(self, prompt_embeds, id_embeds, class_tokens_mask):
        """
        Args:
            prompt_embeds (Tensor): Text encoder embeddings [batch, seq_len, embed_dim]
            id_embeds (Tensor): ID embeddings [batch, max_inputs, 1, embed_dim]
            class_tokens_mask (Tensor): Mask indicating which tokens to replace [batch, seq_len]
        
        Returns:
            Tensor: Updated prompt embeddings.
        """
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        batch_size, max_num_inputs = id_embeds.shape[:2]
        seq_length = prompt_embeds.shape[1]

        num_inputs = class_tokens_mask.sum(dim=1)

        flat_id_embeds = id_embeds.view(-1, id_embeds.shape[-2], id_embeds.shape[-1])
        valid_id_mask = (torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :] < num_inputs[:, None])

        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds_flat = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask_flat = class_tokens_mask.view(-1)

        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])

        image_token_embeds = prompt_embeds_flat[class_tokens_mask_flat]
        stacked_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)

        assert class_tokens_mask_flat.sum() == stacked_embeds.shape[0], (
            f"Mismatch between mask sum and stacked embeds: {class_tokens_mask_flat.sum()} vs {stacked_embeds.shape[0]}"
        )

        prompt_embeds_flat.masked_scatter_(class_tokens_mask_flat[:, None], stacked_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds_flat.view(batch_size, seq_length, -1)

        return updated_prompt_embeds

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    """ID Encoder combining vision features and text prompts."""
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(embed_dim=2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        """
        Args:
            id_pixel_values (Tensor): Images [batch, num_inputs, channels, height, width]
            prompt_embeds (Tensor): Text embeddings [batch, seq_len, embed_dim]
            class_tokens_mask (Tensor): Mask of class tokens to update
        
        Returns:
            Tensor: Updated text embeddings incorporating ID image features.
        """
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        vision_outputs = self.vision_model(id_pixel_values)
        shared_id_embeds = vision_outputs[1]  # Use pooled output

        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)

        combined_id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)

        updated_prompt_embeds = self.fuse_module(prompt_embeds, combined_id_embeds, class_tokens_mask)

        return updated_prompt_embeds

if __name__ == "__main__":
    encoder = PhotoMakerIDEncoder()
    print("PhotoMakerIDEncoder initialized successfully.")