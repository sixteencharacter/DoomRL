import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from env import device , n_actions
from torchinfo import summary
from icecream import ic
from config import *

# Baseline from 1605.02097 (https://arxiv.org/pdf/1605.02097)
class CNN(nn.Module) :

    def __init__(self,n_actions) :
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,7)
        self.conv2 = nn.Conv2d(32,32,4)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.ff = nn.Linear(3072,800)
        self.ff2 = nn.Linear(800,n_actions)
        
    
    def forward(self,x) : 
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.flatten(1)
        x = F.relu(self.ff(x))
        x = self.ff2(x)
        return x

class ResNet(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feat_dim, n_actions)

    def forward(self, x):
        # We now expect x to be pre-resized to 112x112 or similar by the preprocessor
        x = self.backbone(x)
        return self.head(x)

class StarformerQNet(nn.Module):
    """Faithful StARformer (Shang 2022) Q-network.

    Per-step token = state_token + action_token + rtg_token + positional_embed.
    A 6L/8H/192d transformer encoder consumes (B, K, D), and a linear head emits
    (B, K, n_actions). The caller selects the last position for online TD.

    Inputs:
      states:  (B, K, 3, 45, 60) float in [0, 1]
      actions: (B, K) long
      rtgs:    (B, K) float
    """

    def __init__(
        self,
        n_actions: int,
        K: int,
        n_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 192,
        dropout: float = 0.1,
        use_rtg: bool = True,
        rtg_scale: float = 1.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.K = K
        self.d_model = d_model
        self.use_rtg = use_rtg
        self.rtg_scale = rtg_scale

        # Conv stem: 4x4 stride 4 on 3x45x60 -> ~ 11x15 = 165 patch tokens.
        self.conv_stem = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        # Spatial pooling over patch tokens to a single per-step state token.
        self.state_proj = nn.Linear(d_model, d_model)

        self.act_embed = nn.Embedding(n_actions, d_model)
        self.rtg_embed = nn.Linear(1, d_model)

        self.pos_embed = nn.Parameter(torch.zeros(1, K, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.q_head = nn.Linear(d_model, n_actions)

        self._init_weights()

    def _init_weights(self):
        # GPT-style normal(0, 0.02) for embeddings + positional + linear weights.
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.act_embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _state_tokens(self, states: torch.Tensor) -> torch.Tensor:
        # states: (B, K, 3, H, W)
        B, K, C, H, W = states.shape
        x = states.reshape(B * K, C, H, W)
        x = self.conv_stem(x)                # (B*K, D, H', W')
        x = x.flatten(2).mean(dim=2)         # (B*K, D)
        x = self.state_proj(x)
        return x.reshape(B, K, self.d_model)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, K = actions.shape
        assert K == self.K, f"forward got K={K}, expected {self.K}"

        state_tok = self._state_tokens(states)                         # (B,K,D)
        act_tok = self.act_embed(actions.long())                       # (B,K,D)
        if self.use_rtg:
            rtg_in = (rtgs * self.rtg_scale).unsqueeze(-1)             # (B,K,1)
            rtg_tok = self.rtg_embed(rtg_in)                           # (B,K,D)
        else:
            rtg_tok = torch.zeros_like(state_tok)

        tokens = state_tok + act_tok + rtg_tok + self.pos_embed[:, :K, :]

        # attn_mask is bool (B, K): True at valid positions -> convert to key_padding_mask
        key_padding_mask = None
        if attn_mask is not None:
            # PyTorch convention: True means "ignore"
            key_padding_mask = ~attn_mask

        x = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return self.q_head(x)                                           # (B,K,n_actions)

class StarformerActorCritic(nn.Module):
    """STARFORMER backbone shared by actor + critic heads at last sequence position.

    forward(states, actions, rtgs) -> (logits, value)
      states:  (B, K, 3, H, W)
      actions: (B, K) long
      rtgs:    (B, K) float
      logits:  (B, n_actions)
      value:   (B, 1)
    """

    def __init__(
        self,
        n_actions: int,
        K: int,
        n_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 192,
        dropout: float = 0.1,
        use_rtg: bool = True,
        rtg_scale: float = 1.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.K = K
        self.d_model = d_model
        self.use_rtg = use_rtg
        self.rtg_scale = rtg_scale

        self.conv_stem = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        self.state_proj = nn.Linear(d_model, d_model)
        self.act_embed = nn.Embedding(n_actions, d_model)
        self.rtg_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, K, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.actor = nn.Linear(d_model, n_actions)
        self.critic = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.act_embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _state_tokens(self, states: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = states.shape
        x = states.reshape(B * K, C, H, W)
        x = self.conv_stem(x)
        x = x.flatten(2).mean(dim=2)
        x = self.state_proj(x)
        return x.reshape(B, K, self.d_model)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        B, K = actions.shape
        assert K == self.K, f"forward got K={K}, expected {self.K}"

        state_tok = self._state_tokens(states)
        act_tok = self.act_embed(actions.long())
        if self.use_rtg:
            rtg_in = (rtgs * self.rtg_scale).unsqueeze(-1)
            rtg_tok = self.rtg_embed(rtg_in)
        else:
            rtg_tok = torch.zeros_like(state_tok)

        tokens = state_tok + act_tok + rtg_tok + self.pos_embed[:, :K, :]

        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask

        x = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        last = x[:, -1, :]
        logits = self.actor(last)
        value = self.critic(last)
        return logits, value


class ActorCriticResNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.actor = nn.Linear(feat_dim, n_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        x = self.backbone(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.ff = nn.Linear(3072, 800)
        
        self.actor = nn.Linear(800, n_actions)
        self.critic = nn.Linear(800, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.flatten(1)
        x = F.relu(self.ff(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def create_q_network(arch, n_actions):
    if arch == "Baseline":
        return ActorCriticCNN(n_actions=n_actions) if USE_PPO else CNN(n_actions=n_actions)
    if arch == "ResNet":
        return ActorCriticResNet(n_actions=n_actions) if USE_PPO else ResNet(n_actions=n_actions)
    if arch == "STARFORMER":
        from config import (
            STARFORMER_K, STARFORMER_LAYERS, STARFORMER_HEADS,
            STARFORMER_DIM, USE_RTG, STARFORMER_RTG_SCALE,
        )
        cls = StarformerActorCritic if USE_PPO else StarformerQNet
        return cls(
            n_actions=n_actions,
            K=STARFORMER_K,
            n_layers=STARFORMER_LAYERS,
            n_heads=STARFORMER_HEADS,
            d_model=STARFORMER_DIM,
            use_rtg=USE_RTG,
            rtg_scale=STARFORMER_RTG_SCALE,
        )
    raise ValueError(f"Unsupported ARCH: {arch}")

if __name__ == "__main__" :
    model = None
    try :
        model = create_q_network("Baseline", n_actions=2 ** (n_actions))
        print("All model compiled successfully")
    except :
        print(summary(model,(1,3,120,160)))
        print("Error occurred in some model")