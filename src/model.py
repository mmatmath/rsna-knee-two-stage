import torch
import torch.nn as nn
import timm
import transformers
from transformers.models.deberta_v2 import DebertaV2Model


class AttentionPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0]


class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class MaxPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        return last_hidden_state.masked_fill(~mask, -1e4).max(1).values


class GemPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, last_hidden_state, attention_mask):
        x = last_hidden_state.clamp(min=self.eps).pow(self.p)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return x.pow(1.0 / self.p)


def get_pooling(name):
    if name == "cls":
        return AttentionPooling()
    if name == "gem":
        return GemPooling()
    if name == "max":
        return MaxPooling()
    return MeanPooling()


class Stage1(nn.Module):
    def __init__(self, encoder_name="convnext_base.dinov3_lvd1689m", pretrained=True,
                 n_classes=12, n_slots=6, gc=False):
        super().__init__()
        self.image_encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
        if gc and hasattr(self.image_encoder, "set_grad_checkpointing"):
            self.image_encoder.set_grad_checkpointing(True)
        self.feature_dim = self.image_encoder.num_features
        self.slot_embedding = nn.Embedding(n_slots, self.feature_dim)
        self.query = nn.Parameter(torch.randn(n_classes, self.feature_dim) * self.feature_dim ** -0.5)
        self.out_weight = nn.Parameter(torch.randn(n_classes, self.feature_dim) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, images, token_mask, slot_ids):
        batch, tokens, channels, height, width = images.shape
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        x = images.reshape(batch * tokens, channels, height, width)
        features = self.image_encoder(x)
        # [B*T, D] -> [B, T, D]
        features = features.reshape(batch, tokens, -1)
        h = features + self.slot_embedding(slot_ids)

        attention = torch.einsum("btd,od->bot", h, self.query) / self.feature_dim ** 0.5
        attention = attention.masked_fill(token_mask.unsqueeze(1) == 0, -1e4)
        attention = attention.softmax(-1)
        context = torch.einsum("bot,btd->bod", attention, h)
        logits = (context * self.out_weight.unsqueeze(0)).sum(-1) + self.out_bias
        return logits, features, attention


class Stage2(nn.Module):
    def __init__(self, transformer_name="microsoft/deberta-v3-base", hidden_size=1024,
                 intermediate_size=1024, attention_heads=8, num_hidden_layers=3,
                 attention_dropout=0.05, hidden_dropout=0.15,
                 classifier_dropout=0.15, n_classes=12, n_slots=6, gc=False, pool="gem"):
        super().__init__()
        config = transformers.AutoConfig.from_pretrained(transformer_name)
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.vocab_size = 3
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = attention_heads
        config.attention_probs_dropout_prob = attention_dropout
        config.hidden_dropout_prob = hidden_dropout
        config.hidden_act = "gelu"
        config.conv_act = "gelu"
        config.conv_kernel_size = 3
        self.config = config
        self.transformer = transformers.AutoModel.from_config(config)
        if gc:
            self.transformer.gradient_checkpointing_enable()

        scale = hidden_size ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
        self.slot_embedding = nn.Embedding(n_slots, hidden_size)
        self.pool = get_pooling(pool)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, features, slot_ids, attention_mask):
        batch = features.shape[0]
        x = features + self.slot_embedding(slot_ids)
        cls = self.cls_embedding.repeat(batch, 1, 1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones((batch, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        x = self.transformer(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        x = torch.cat([self.pool(x, attention_mask), x[:, 0]], dim=-1)
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, encoder_name, transformer_config, classifier_dropout=0.15,
                 n_classes=12, n_slots=6, pool="gem"):
        super().__init__()
        self.image_encoder = timm.create_model(encoder_name, pretrained=False, num_classes=0)
        self.transformer = DebertaV2Model(transformer_config)
        hidden_size = transformer_config.hidden_size
        self.slot_embedding = nn.Embedding(n_slots, hidden_size)
        self.cls_embedding = nn.Parameter(hidden_size ** -0.5 * torch.randn(1, 1, hidden_size))
        self.pool = get_pooling(pool)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(), nn.Dropout(classifier_dropout), nn.Linear(hidden_size, n_classes),
        )

    def forward_encoder(self, images):
        return self.image_encoder(images)

    def forward_transformer(self, features, slot_ids, attention_mask):
        batch = features.shape[0]
        x = features + self.slot_embedding(slot_ids)
        x = torch.cat([self.cls_embedding.repeat(batch, 1, 1), x], dim=1)
        cls_mask = torch.ones((batch, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        x = self.transformer(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        x = torch.cat([self.pool(x, attention_mask), x[:, 0]], dim=-1)
        return self.fc(x)

    def forward(self, images, slot_ids, attention_mask):
        features = self.forward_encoder(images)
        return self.forward_transformer(features, slot_ids, attention_mask)
