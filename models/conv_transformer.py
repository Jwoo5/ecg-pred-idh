import logging

import torch
import torch.nn as nn

import utils
from modules import (
    ConvFeatureExtraction,
    TransformerEncoder,
    LayerNorm,
    GradMultiply,
)

logger = logging.getLogger(__name__)

class ConvTransformerModel(nn.Module):
    def __init__(
        self,
        conv_feature_layers="[(256, 3, 2)] * 4 + [(256, 2, 2)] * 2",
        in_d=12,
        conv_bias=False,
        feature_grad_mult=1.0,
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=12,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.1,
        encoder_layerdrop=0.0,
        dropout_input=0.1,
        dropout_features=0.0,
        conv_pos=128,
        conv_pos_groups=16,
        layer_norm_first=False,
        num_labels=1,
        **kwargs,
    ):
        super().__init__()
        self.conv_feature_layers = conv_feature_layers
        feature_enc_layers = eval(conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=in_d,
            dropout=0.0,
            mode="default",
            conv_bias=conv_bias
        )

        self.encoder_embed_dim = encoder_embed_dim

        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim
            else None
        )

        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)
        
        self.feature_grad_mult = feature_grad_mult
        self.final_dim = encoder_embed_dim

        self.encoder = TransformerEncoder(
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            encoder_embed_dim= encoder_embed_dim,
            conv_pos=conv_pos,
            conv_pos_groups=conv_pos_groups,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            layer_norm_first=layer_norm_first
        )
        self.layer_norm = LayerNorm(self.embed)

        self.final_proj = nn.Linear(encoder_embed_dim, num_labels)
        nn.init.xavier_uniform_(self.final_proj.weight)
        nn.init.constant_(self.final_proj.bias, 0.0)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        
        conv_cfg_list = eval(self.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])
        
        return input_lengths.to(torch.long)
    
    def forward(
        self,
        source,
        padding_mask=None,
        mask=False,
        mask_indices=None,
        **kwargs
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        
        features = features.transpose(1,2)
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:,0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype = features.dtype, device = features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device = padding_mask.device),
                    output_lengths - 1
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        x = self.dropout_input(features)

        x = self.encoder(x, padding_mask=padding_mask)

        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.final_proj(x)

        return {"x": x, "padding_mask": padding_mask}
    
    def get_logits(self, net_output, normalize=False):
        logits = net_output["x"].float()

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits.squeeze(-1)
    
    def get_targets(self, sample, net_output):
        return sample["label"].float()