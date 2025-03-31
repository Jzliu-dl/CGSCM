import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class maskPredictor(nn.Module):
    def __init__(self, d_model):
        super(maskPredictor, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

        self.mask_generator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, image_features, text_features, image_padding_mask=None, text_padding_mask=None):

        queries = self.query_proj(text_features)
        keys = self.key_proj(image_features)

        keys = rearrange(keys, 'b h w d -> b (h w) d')
        image_padding_mask = rearrange(image_padding_mask, 'b h w -> b (h w)')

        #attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale # (batch_size, l, h*w)

        attention_scores = F.cosine_similarity(queries.unsqueeze(2), keys.unsqueeze(1), dim=-1) # (batch_size, L_text, L_image)
        #print(attention_scores.shape)

        
        if image_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(image_padding_mask.unsqueeze(1), float('-inf'))
  
        max_weights, _ = attention_scores.max(dim=-1)  # 最大注意力权重 (batch_size, L_text)
        sum_attention_scores_unsq = max_weights.unsqueeze(-1)
        continuous_mask = self.mask_generator(sum_attention_scores_unsq).squeeze(-1)

        return continuous_mask


        '''

        sum_attention_scores = attention_scores.masked_fill(attention_scores == float('-inf'), 0).sum(dim=-1)
        valid_counts = (attention_scores != float('-inf')).sum(dim=-1)
        sum_attention_scores = sum_attention_scores / valid_counts.clamp(min=1)

        sum_attention_scores_unsq = sum_attention_scores.unsqueeze(-1)
        continuous_mask = self.mask_generator(sum_attention_scores_unsq).squeeze(-1)

        return continuous_mask
        '''

