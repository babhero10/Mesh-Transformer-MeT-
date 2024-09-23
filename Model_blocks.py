import torch
import torch.nn as nn
    
class OneLayerFeedForward(nn.Module):
    def __init__(self, dim_in, dim_out, add_norm=False, add_residual=False, dropout=0.1):
        super().__init__()
        self.add_norm = add_norm
        self.add_residual = add_residual
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Optional layer normalization
        if add_norm:
            self.layer_norm = nn.LayerNorm(dim_in)
        
        # Feedforward one-layer neural network
        self.ff_one_layer = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features):
        normalized_feature = features

        # Apply normalization if requested
        if self.add_norm:
            normalized_feature = self.layer_norm(features)

        # Apply feedforward layer
        out = self.ff_one_layer(normalized_feature)
        
        # Add residual connection if requested, make sure input/output dimensions match
        if self.add_residual and self.dim_in == self.dim_out:
            out = out + features
        
        return out


class OutputFeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.1):
        super().__init__()
        
        self.ff_out_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, features):
        return self.ff_out_layer(features)
    
    
class SACluster(nn.Module):
    def __init__(self, P_dim, num_heads, dropout=0.1, bias=True, add_norm=True, residual=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(P_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.add_norm = add_norm
        self.residual = residual
        
        if self.add_norm:
            self.layer_norm = nn.LayerNorm(P_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, P_feature):
        P_feature_norm = P_feature
        
        if self.add_norm:
            P_feature_norm = self.layer_norm(P_feature)
            
        attn_output, _ = self.multihead_attn(
            P_feature_norm, P_feature_norm, P_feature_norm, 
        )
        
        attn_output = self.dropout(attn_output)

        if self.residual:
            attn_output = attn_output + P_feature

        return attn_output
    
    
class SATriangle(nn.Module):
    def __init__(self, E_dim, num_heads, dropout=0.1, bias=True, add_norm=True, residual=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(E_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.add_norm = add_norm
        self.residual = residual
        
        if self.add_norm:
            self.layer_norm = nn.LayerNorm(E_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_feature, A_hat):
        E_feature_norm = E_feature
        
        if self.add_norm:
            E_feature_norm = self.layer_norm(E_feature)
            
        attn_output, _ = self.multihead_attn(
            E_feature_norm, E_feature_norm, E_feature_norm,
            attn_mask = A_hat 
        )
        
        attn_output = self.dropout(attn_output)

        if self.residual:
            attn_output = attn_output + E_feature

        return attn_output
        
        
class ClusterTriangleBlock(nn.Module):
    def __init__(self, P_dim, E_dim, num_heads, dropout=0.1, bias=True, add_norm=True, residual=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(P_dim, num_heads, dropout=dropout, bias=bias, kdim=E_dim, vdim=E_dim, batch_first=True)
        self.add_norm = add_norm
        self.residual = residual
        self.num_head = num_heads
        if self.add_norm:
            self.layer_norm = nn.LayerNorm(P_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, P_features, E_features, C_hat):

        P_feature_norm = P_features
        
        if self.add_norm:
            P_feature_norm = self.layer_norm(P_features)
        
        attn_output, _ = self.multihead_attn(
            P_feature_norm, E_features, E_features,
            attn_mask=C_hat 
        )
        
        attn_output = self.dropout(attn_output)

        if self.residual:
            attn_output = attn_output + P_features

        return attn_output
    
    
class TriangleClusterBlock(nn.Module):
    def __init__(self, E_dim, P_dim, dropout=0.1, add_norm=True):
        super().__init__()
        self.add_norm = add_norm
        
        # Feedforward layer for projection using OneLayerFeedForward
        self.ff_proj = OneLayerFeedForward(P_dim, E_dim, add_norm=False, add_residual=False, dropout=dropout)
        
        if add_norm:
            self.layer_norm = nn.LayerNorm(E_dim)
        
    def forward(self, E_features, C_matrix, P_features):
        # Apply normalization if needed
        E_norm = E_features
        if self.add_norm:
            E_norm = self.layer_norm(E_features)
        
        # Compute average cluster token for each triangle
        cluster_avg = C_matrix @ P_features  # [batch_size, num_triangles, P_dim]
        
        # Project the average cluster representation using OneLayerFeedForward
        projected_avg = self.ff_proj(cluster_avg)  # [batch_size, num_triangles, E_dim]
        
        # Update triangle representations
        out = E_norm + projected_avg
        
        return out
    

class EncoderLayer(nn.Module):
    def __init__(self, P_dim, E_dim, dropout=0.1, num_heads=8, add_norm=True, residual=True):
        super().__init__()
        
        self.CT = ClusterTriangleBlock(P_dim, E_dim, num_heads=num_heads, dropout=dropout, bias=False, add_norm=add_norm, residual=residual)
        self.TC = TriangleClusterBlock(E_dim, P_dim, dropout=dropout, add_norm=add_norm)
        self.SA_T = SATriangle(E_dim, num_heads=num_heads, dropout=dropout, bias=False, add_norm=add_norm, residual=residual)
        self.SA_C = SACluster(P_dim, num_heads=num_heads, dropout=dropout, bias=False, add_norm=add_norm, residual=residual)
        self.ff_SA_T = OneLayerFeedForward(E_dim, E_dim, add_norm=True, add_residual=True, dropout=dropout)
        self.ff_SA_C = OneLayerFeedForward(P_dim, P_dim, add_norm=True, add_residual=True, dropout=dropout)
        
    def forward(self, E_features, P_features, A_hat, C_hat, C_matrix):
        TC_out = self.TC(E_features, C_matrix, P_features)

        CT_out = self.CT(P_features, E_features, C_hat) 
        
        SATriangle_out = self.SA_T(TC_out, A_hat) 

        SACluster_out = self.SA_C(CT_out)
        
        ff_SA_T_out = self.ff_SA_T(SATriangle_out)
        
        ff_SA_C_out = self.ff_SA_C(SACluster_out)
        
        return ff_SA_T_out, ff_SA_C_out
    

class MeshTransformer(nn.Module):
    def __init__(self, t_dim, J_dim, P_dim, E_dim, num_of_labels=3, num_of_encoder_layers=2, dropout=0.1, num_heads=8, add_norm=True, residual=True):
        super().__init__()
        
        self.embedded_layer = nn.Embedding(J_dim, P_dim)
        self.ff_t_in = OneLayerFeedForward(t_dim, E_dim, dropout=dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(P_dim, E_dim, dropout=dropout, num_heads=num_heads, add_norm=add_norm, residual=residual)
            for _ in range(num_of_encoder_layers)
        ])        
        
        self.num_heads = num_heads
        
        self.ff_s_out = OutputFeedForward(E_dim, E_dim//2, num_of_labels)
        
    def forward(self, T_features, J_features, A_matrix, C_matrix):
        A_hat = A_matrix.to(torch.bool).repeat(self.num_heads, 1, 1)
        C_hat = C_matrix.to(torch.bool).repeat(self.num_heads, 1, 1)
        
        E_features = self.ff_t_in(T_features)
        P_features = self.embedded_layer(torch.argmax(J_features, dim=-1))

        for encoder_layer in self.encoder_layers:
            E_features, P_features = encoder_layer(E_features, P_features, A_hat, C_hat, C_matrix)
        
        S_out = self.ff_s_out(E_features)
        
        return S_out, None