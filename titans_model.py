import torch
import torch.nn as nn
import math

class LongTermMemory(nn.Module):
    """Neural memory module that learns to memorize important information."""
    def __init__(self, dim, depth=2):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Deep MLP for memory processing
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU()
            ])
        self.memory_network = nn.Sequential(*layers)
        
        # Parameters for memory updating
        self.theta = nn.Parameter(torch.ones(1))  # Learning rate
        self.eta = nn.Parameter(torch.ones(1))    # Momentum rate
        self.alpha = nn.Parameter(torch.ones(1))  # Forget rate

    def forward(self, x, prev_memory=None, prev_surprise=None):
        batch_size, seq_len, _ = x.shape
        print(f"theta: {self.theta.item()}, eta: {self.eta.item()}, alpha: {self.alpha.item()}")
       
        # Initialize memory if needed
        if prev_memory is None:
            prev_memory = torch.zeros_like(x)
        if prev_surprise is None:
            prev_surprise = torch.zeros_like(x)
        
        # Process each sequence position through memory network
        memory_network_out = []
        for i in range(seq_len):
            pos_out = self.memory_network(x[:, i])
            memory_network_out.append(pos_out)
        
        # Stack outputs back into sequence
        memory_network_out = torch.stack(memory_network_out, dim=1)
        
        # Calculate momentary surprise
        momentary_surprise = memory_network_out - prev_memory

        # Update surprise with momentum
        surprise = (self.eta * prev_surprise -
                    self.theta * momentary_surprise)

        # Update memory with forget mechanism
        memory = (1 - self.alpha) * prev_memory + surprise

        # Clamp memory and surprise
        memory = torch.clamp(memory, min=-1e5, max=1e5)
        surprise = torch.clamp(surprise, min=-1e5, max=1e5)

        # Clamp alpha
        self.alpha.data.clamp_(0.001, 0.1)

        print(f"theta: {self.theta.item()}, eta: {self.eta.item()}, alpha: {self.alpha.item()}")

        return memory, surprise


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5 + 1e-8  # Add epsilon here

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TitansMAC(nn.Module):
    """Memory as Context (MAC) variant of Titans."""
    def __init__(self, dim, num_heads=8, memory_depth=2, context_size=512):
        super().__init__()
        self.dim = dim
        self.context_size = context_size
        
        # Memory components
        self.memory = LongTermMemory(dim, depth=memory_depth)
        self.memory_query = nn.Linear(dim, dim)
        
        # Attention components
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Persistent memory (learnable tokens)
        self.persistent_memory = nn.Parameter(torch.randn(8, dim))

    def forward(self, x, prev_memory=None, prev_surprise=None):
        B = x.shape[0]
        
        # Update long-term memory
        memory, surprise = self.memory(x, prev_memory, prev_surprise)
        
        # Query memory
        memory_query = self.memory_query(x)
        memory_context = memory * memory_query
        
        # Combine input with memory and persistent memory
        persistent = self.persistent_memory.expand(B, -1, -1)
        context = torch.cat([persistent, memory_context, x], dim=1)
        
        # Apply attention
        attended = self.attention(self.norm1(context))
        x = context + attended
        
        # Apply FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, memory, surprise