"""GADAN: Geometry-Aware Decoupled Attention Network."""

import copy
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (
    RobertaTokenizerFast,
    BertTokenizerFast,
    RobertaModel,
    BertModel,
)

from util.misc import NestedTensor, nested_tensor_from_videos_list, inverse_sigmoid

from .base_model import MLP, RobertaPoolout, FeatureResizer, _get_clones
from .backbone import build_backbone
from .criterion import SetCriterion
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .position_encoding import PositionEmbeddingSine1D
from .postprocessors import build_postprocessors
from .segmentation import VisionLanguageFusionModule
from .gsbi import GSBICrossAttention, generate_vis_coords

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
#  Text Encoders
# ---------------------------------------------------------------------------

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, pad_token_id, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bilstm_output_dim = hidden_dim * 2
        self.proj = nn.Linear(bilstm_output_dim, output_dim) if bilstm_output_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=input_ids.shape[1])
        outputs = self.proj(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        return outputs


class RobertaTextEncoder(nn.Module):
    """Wraps HuggingFace RobertaModel with a FeatureResizer to match d_model."""

    def __init__(self, model_name, output_dim, freeze=False):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.resizer = FeatureResizer(
            input_feat_size=self.encoder.config.hidden_size,
            output_feat_size=output_dim,
            dropout=0.1,
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, tokenized):
        encoded = self.encoder(**tokenized)
        return self.resizer(encoded.last_hidden_state)


class BertTinyTextEncoder(nn.Module):
    """Wraps a tiny BERT model (e.g. google/bert_uncased_L-2_H-128_A-2) with a FeatureResizer."""

    def __init__(self, model_name, output_dim):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.resizer = FeatureResizer(
            input_feat_size=self.encoder.config.hidden_size,
            output_feat_size=output_dim,
            dropout=0.1,
        )

    def forward(self, tokenized):
        encoded = self.encoder(**tokenized)
        return self.resizer(encoded.last_hidden_state)


# ---------------------------------------------------------------------------
#  Tokenizer factory
# ---------------------------------------------------------------------------

def build_tokenizer(text_encoder_type, tokenizer_path="roberta-base"):
    if text_encoder_type in ("roberta", "roberta_frozen", "bilstm"):
        return RobertaTokenizerFast.from_pretrained(tokenizer_path)
    elif text_encoder_type == "bert_tiny":
        return BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")


# ---------------------------------------------------------------------------
#  GSBI Fusion Wrapper
#  Thin adapter that exposes the same call signature as VisionLanguageFusionModule
#  but routes the cross-attention through GSBICrossAttention with geometric bias.
# ---------------------------------------------------------------------------

class GSBIFusionWrapper(nn.Module):
    """
    Wraps GSBICrossAttention to match the VisionLanguageFusionModule interface
    (sequence-first format + position embeddings + gating).

    Handles:
      1. pos-embed addition to Q / K
      2. sequence-first (L,B,C) → batch-first (B,L,C) conversion
      3. GSBICrossAttention call (with vis_coords & masks)
      4. batch-first → sequence-first conversion back
      5. element-wise gating:  tgt = tgt * tgt2
    """

    def __init__(self, d_model, nhead, d_geo=64, dropout=0.0):
        super().__init__()
        self.gsbi_attn = GSBICrossAttention(
            d_model=d_model, nhead=nhead, d_geo=d_geo, dropout=dropout
        )

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, vis_coords,
                memory_key_padding_mask=None, pos=None, query_pos=None,
                text_mask=None):
        """
        Args:
            tgt:                     (L_t, B, d_model) — text features, sequence-first
            memory:                  (L_v, B, d_model) — visual features, sequence-first
            vis_coords:              (B, L_v, 4)       — normalised bbox coords
            memory_key_padding_mask: (B, L_v) | None   — visual padding, True = masked
            pos:                     (L_v, B, d_model) — visual position encoding
            query_pos:               (L_t, B, d_model) — text position encoding (unused here)
            text_mask:               (B, L_t) | None   — text padding, True = masked

        Returns:
            tgt: (L_t, B, d_model) — gated text features, sequence-first
        """
        # 1. add position embeddings
        q = self.with_pos_embed(tgt, query_pos)      # (L_t, B, d_model)
        k = self.with_pos_embed(memory, pos)          # (L_v, B, d_model)

        # 2. sequence-first → batch-first
        q_bf = q.transpose(0, 1)                      # (B, L_t, d_model)
        k_bf = k.transpose(0, 1)                      # (B, L_v, d_model)
        v_bf = memory.transpose(0, 1)                 # (B, L_v, d_model)

        # 3. GSBI cross-attention  (batch-first in, batch-first out)
        tgt2_bf = self.gsbi_attn(
            query=q_bf,
            key=k_bf,
            value=v_bf,
            vis_coords=vis_coords,
            text_mask=text_mask,
            key_padding_mask=memory_key_padding_mask,
        )  # (B, L_t, d_model)

        # 4. batch-first → sequence-first
        tgt2 = tgt2_bf.transpose(0, 1)                # (L_t, B, d_model)

        # 5. element-wise gating (same as original VisionLanguageFusionModule)
        tgt = tgt * tgt2
        return tgt


# ---------------------------------------------------------------------------
#  Main Model
# ---------------------------------------------------------------------------

class GADAN(nn.Module):
    """
    GADAN: Geometry-Aware Decoupled Attention Network.

    Configurable text encoder (BiLSTM / RoBERTa / BERT-tiny) +
    deformable DETR-style visual encoder-decoder,
    with geometry-aware spatial bias injection (GSBI) replacing standard
    text→visual cross-attention fusion.
    """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, args=None):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage is False, "args.two_stage must be false!"

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # ---- text encoder selection ----
        text_encoder_type = getattr(args, "text_encoder_type", "bilstm")
        self.text_encoder_type = text_encoder_type

        if text_encoder_type == "bilstm":
            tokenizer_path = getattr(args, "tokenizer_path", "roberta-base")
        elif text_encoder_type == "bert_tiny":
            tokenizer_path = getattr(args, "tokenizer_path", "google/bert_uncased_L-2_H-128_A-2")
        else:
            tokenizer_path = getattr(args, "tokenizer_path", "roberta-base")
        self.tokenizer = build_tokenizer(text_encoder_type, tokenizer_path)

        if text_encoder_type == "bilstm":
            bilstm_embed_dim = getattr(args, "bilstm_embed_dim", 300)
            bilstm_hidden_dim = getattr(args, "bilstm_hidden_dim", hidden_dim // 2)
            bilstm_num_layers = getattr(args, "bilstm_num_layers", 2)
            bilstm_dropout = getattr(args, "bilstm_dropout", 0.1)
            self.text_encoder = BiLSTMTextEncoder(
                vocab_size=self.tokenizer.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
                output_dim=hidden_dim,
                embed_dim=bilstm_embed_dim,
                hidden_dim=bilstm_hidden_dim,
                num_layers=bilstm_num_layers,
                dropout=bilstm_dropout,
            )
        elif text_encoder_type == "roberta":
            text_encoder_path = getattr(args, "text_encoder_path", "roberta-base")
            self.text_encoder = RobertaTextEncoder(
                model_name=text_encoder_path,
                output_dim=hidden_dim,
                freeze=False,
            )
        elif text_encoder_type == "roberta_frozen":
            text_encoder_path = getattr(args, "text_encoder_path", "roberta-base")
            self.text_encoder = RobertaTextEncoder(
                model_name=text_encoder_path,
                output_dim=hidden_dim,
                freeze=True,
            )
        elif text_encoder_type == "bert_tiny":
            text_encoder_path = getattr(args, "text_encoder_path", "google/bert_uncased_L-2_H-128_A-2")
            self.text_encoder = BertTinyTextEncoder(
                model_name=text_encoder_path,
                output_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")

        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(False)

        # ---- GSBI: geometry-aware text→visual cross-attention ----
        d_geo = getattr(args, 'gsbi_d_geo', 64)
        self.fusion_module_text = GSBIFusionWrapper(
            d_model=hidden_dim, nhead=8, d_geo=d_geo, dropout=0.0
        )
        # ---- standard visual→text cross-attention (unchanged) ----
        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, samples: NestedTensor, captions, targets):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)

        features, pos = self.backbone(samples)

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor(
                [i * t + target['valid_indices'] for i, target in enumerate(targets)]
            ).to(pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)
        text_word_initial_features = text_word_features

        # ---- iterate over backbone feature levels ----
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

            # ---- GSBI: generate vis_coords for this spatial level ----
            vis_coords = generate_vis_coords(h, w, t, b, pos[0].device)  # (b, t*h*w, 4)

            # ---- GSBI: text → visual cross-attention with geometry bias ----
            text_word_features = self.fusion_module_text(
                tgt=text_word_features,
                memory=src_proj_l,
                vis_coords=vis_coords,
                memory_key_padding_mask=mask,
                pos=pos_l,
                query_pos=None,
                text_mask=text_word_masks,
            )

            # ---- standard: visual → text cross-attention (unchanged) ----
            src_proj_l = self.fusion_module(
                tgt=src_proj_l,
                memory=text_word_initial_features,
                memory_key_padding_mask=text_word_masks,
                pos=text_pos,
                query_pos=None,
            )
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        # ---- extra feature levels (if num_feature_levels > backbone outputs) ----
        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                # GSBI: generate vis_coords for this extra level
                vis_coords = generate_vis_coords(h, w, t, b, pos[0].device)

                text_word_features = self.fusion_module_text(
                    tgt=text_word_features,
                    memory=src,
                    vis_coords=vis_coords,
                    memory_key_padding_mask=mask,
                    pos=pos_l,
                    query_pos=None,
                    text_mask=text_word_masks,
                )
                src = self.fusion_module(
                    tgt=src,
                    memory=text_word_initial_features,
                    memory_key_padding_mask=text_word_masks,
                    pos=text_pos,
                    query_pos=None,
                )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        query_embeds = self.query_embed.weight
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds)

        out = {}
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward_text(self, captions, device):
        if not isinstance(captions[0], str):
            raise ValueError("Please mask sure the caption is a list of string")

        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        ).to(device)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        if self.text_encoder_type == "bilstm":
            text_features = self.text_encoder(tokenized.input_ids, tokenized.attention_mask)
        else:
            text_features = self.text_encoder(tokenized)

        return NestedTensor(text_features, text_attention_mask)


# ---------------------------------------------------------------------------
#  Build function
# ---------------------------------------------------------------------------

def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91
    device = torch.device(args.device)

    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = GADAN(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        args=args,
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha,
    )
    criterion.to(device)

    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
