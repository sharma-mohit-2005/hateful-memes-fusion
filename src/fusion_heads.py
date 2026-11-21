"""Fusion heads on top of frozen CLIP embeddings. This is the core of the fusion ablation.

Notation: i = projected image embedding, t = projected text embedding (both d-dim after projection).

  image_only      z = i
  text_only       z = t
  concat          z = [i ; t]                              (early fusion)
  product         z = i * t                                (element-wise interaction only)
  concat_product  z = [i ; t ; i*t ; |i - t|]              (concat + interaction)
  cross           z = flatten(i_small outer t_small)       (Hate-CLIPper style feature interaction matrix)
  gated           g = sigmoid(W[i;t]);  z = g*i + (1-g)*t  (learned per-dimension modality weighting)
  attention       2-layer Transformer over tokens [CLS, i, t] -> CLS  (cross-modal attention)

How it works (one forward pass):
  1. Input: cached CLIP embeddings img [B,768] and txt [B,768] (ViT-L/14).
  2. L2-normalize each vector (unit length), as CLIP itself does before comparing image and text.
  3. Project each modality with its own small MLP (Linear -> GELU -> Dropout) to d dims:
     d = 512 for most fusions, d = 64 for "cross" (so the outer product stays 64*64 = 4096 features).
  4. Combine i and t with the chosen fusion operator -> fused vector z.
  5. Classifier MLP (LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear) maps z to ONE logit.
     sigmoid(logit) = probability that the meme is hateful. Trained with BCEWithLogitsLoss.
  Only these small heads are trained; CLIP stays frozen. Comparing fusions with everything else fixed
  tells us how much the WAY of combining image and text matters (hateful memes need both modalities:
  a harmless picture + harmless text can be hateful together).
"""
# Core PyTorch tensor library.
import torch
# nn: neural-network building blocks (Linear, LayerNorm, TransformerEncoder, ...).
import torch.nn as nn
# F: functional ops without parameters (here F.normalize).
import torch.nn.functional as F

# List of all supported fusion names; train_head.py uses it as the allowed choices for --fusion.
FUSIONS = ["image_only", "text_only", "concat", "product", "concat_product", "cross", "gated", "attention"]


# ---------------------------------------------------------------------------------------------------
# projector
# Purpose : Small per-modality MLP that maps a CLIP embedding into the head's working space.
# Inputs  : in_dim (768 for ViT-L/14), out_dim (d = 512, or 64 for cross), dropout probability.
# Outputs : nn.Sequential module: [B,in_dim] -> [B,out_dim].
# Viva tip: Q: Why project at all if CLIP is already good? A: The trainable projection adapts the
#           general CLIP space to the hate-detection task, and lets us choose d (e.g. 64 for cross).
# Viva tip: Q: Why GELU? A: Smooth version of ReLU (x * Phi(x)); standard in Transformers incl. CLIP.
# ---------------------------------------------------------------------------------------------------
def projector(in_dim, out_dim, dropout):
    # Linear: W x + b; GELU: non-linearity; Dropout: randomly zero activations during training (regularizer).
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))


# ---------------------------------------------------------------------------------------------------
# FusionHead
# Purpose : The full trainable model: normalize -> project -> fuse -> classify, for one fusion type.
# Inputs  : fusion (name from FUSIONS), in_dim (CLIP dim, 768), proj_dim (512), cross_dim (64),
#           hidden (classifier hidden size, 512), dropout (0.3).
# Outputs : forward(img [B,768], txt [B,768]) -> logits [B] (one raw score per meme).
# Viva tip: Q: Why output 1 logit and not 2 classes? A: Binary task; one logit + sigmoid gives
#           P(hateful). It pairs with BCEWithLogitsLoss, which is numerically stable.
# Viva tip: Q: Which fusion is Hate-CLIPper? A: "cross": the outer product i t^T, i.e. a feature
#           interaction matrix where every image dimension is multiplied with every text dimension.
# ---------------------------------------------------------------------------------------------------
class FusionHead(nn.Module):
    # Build all layers needed for the chosen fusion.
    def __init__(self, fusion, in_dim, proj_dim=512, cross_dim=64, hidden=512, dropout=0.3):
        # Initialize nn.Module internals (parameter registration etc.).
        super().__init__()
        # Fail early with a clear message if an unknown fusion name is passed.
        assert fusion in FUSIONS, f"unknown fusion {fusion}"
        # Remember the fusion type; fuse() branches on it.
        self.fusion = fusion
        # Working dim d: 64 for cross (outer product gives d*d = 4096 features; with 512 it would be
        # 262,144 features -> huge classifier, overfitting), otherwise 512.
        d = cross_dim if fusion == "cross" else proj_dim
        # Separate projector for the image embedding: [B,768] -> [B,d].
        self.img_proj = projector(in_dim, d, dropout)
        # Separate projector for the text embedding: [B,768] -> [B,d] (own weights, modality-specific).
        self.txt_proj = projector(in_dim, d, dropout)

        # Gated fusion needs a gate layer: input [i;t] (2d) -> one gate value per dimension (d).
        if fusion == "gated":
            self.gate = nn.Linear(2 * d, d)
        # Attention fusion needs a CLS token, modality embeddings and a small Transformer encoder.
        if fusion == "attention":
            # Learnable [CLS] token [1,1,d] (starts at zero); its output summarizes the whole sequence,
            # like the CLS token in BERT/ViT.
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            # Learnable "modality / position" embeddings for the 3 tokens [CLS, image, text], shape [1,3,d].
            # Small random init (std 0.02, as in BERT/ViT). They tell the Transformer which token is which,
            # because self-attention by itself is order/identity-agnostic.
            self.mod_emb = nn.Parameter(torch.randn(1, 3, d) * 0.02)
            # One Transformer layer: multi-head self-attention (8 heads, each d/8 = 64 dims) + feed-forward
            # (d -> 2d -> d). batch_first: tensors are [B, seq, d]. norm_first=True = pre-norm
            # (LayerNorm before attention/FFN), which trains more stably than post-norm.
            layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=dropout,
                                               batch_first=True, norm_first=True)
            # Stack 2 such layers. enable_nested_tensor=False: avoids a PyTorch fast-path warning with
            # pre-norm layers (we have no padding anyway, all sequences have length 3).
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

        # Size of the fused vector z for each fusion:
        #   image_only/text_only/product/gated/attention -> d; concat -> 2d;
        #   concat_product -> 4d ([i; t; i*t; |i-t|]); cross -> d*d (64*64 = 4096).
        z_dim = {"image_only": d, "text_only": d, "concat": 2 * d, "product": d,
                 "concat_product": 4 * d, "cross": d * d, "gated": d, "attention": d}[fusion]
        # Classifier MLP: LayerNorm(z) rescales features to mean 0 / var 1 (different fusions produce very
        # different scales, e.g. products are tiny), Dropout regularizes, Linear z_dim -> hidden, GELU,
        # Dropout, final Linear hidden -> 1 logit.
        self.classifier = nn.Sequential(nn.LayerNorm(z_dim), nn.Dropout(dropout), nn.Linear(z_dim, hidden),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    # -----------------------------------------------------------------------------------------------
    # fuse
    # Purpose : Turn the two raw CLIP embeddings into one fused vector z using the chosen operator.
    # Inputs  : img [B,768], txt [B,768].
    # Outputs : z [B,z_dim].
    # Viva tip: Q: Why F.normalize? A: CLIP is trained with cosine similarity, so the direction of the
    #           vector carries the meaning, not its length. Unit length puts image and text on the same
    #           scale, so neither modality dominates just because its numbers are bigger.
    # Viva tip: Q: Difference between concat and product? A: concat just places features side by side
    #           (no interaction - the classifier must learn it); product multiplies matching dimensions,
    #           so it directly encodes agreement/interaction between image and text.
    # -----------------------------------------------------------------------------------------------
    def fuse(self, img, txt):
        # L2-normalize the image vector (divide by its Euclidean norm) then project: i is [B,d].
        i = self.img_proj(F.normalize(img, dim=-1))
        # Same for the text vector: t is [B,d].
        t = self.txt_proj(F.normalize(txt, dim=-1))
        # Short alias for the fusion name.
        f = self.fusion
        # Unimodal baseline: ignore text completely, z = i.
        if f == "image_only":
            return i
        # Unimodal baseline: ignore image completely, z = t.
        if f == "text_only":
            return t
        # Early fusion by concatenation: z = [i ; t], shape [B,2d]. No explicit interaction.
        if f == "concat":
            return torch.cat([i, t], -1)
        # Element-wise (Hadamard) product: z_k = i_k * t_k, shape [B,d]. Large where both modalities
        # agree in a dimension; captures interaction, but loses the individual unimodal information.
        if f == "product":
            return i * t
        # Rich combination [B,4d]: i and t (unimodal info), i*t (agreement / interaction),
        # |i - t| (difference / disagreement, useful when image and text contradict each other).
        if f == "concat_product":
            return torch.cat([i, t, i * t, (i - t).abs()], -1)
        # Hate-CLIPper cross fusion: outer product per sample, P[b,j,k] = i[b,j] * t[b,k], shape [B,64,64].
        # Every image dimension interacts with EVERY text dimension (not only the matching one as in
        # product). flatten(1) turns it into [B,4096] for the classifier.
        if f == "cross":
            return torch.einsum("bi,bj->bij", i, t).flatten(1)
        # Gated fusion: gate g in (0,1)^d computed from both modalities via sigmoid(W[i;t] + b).
        if f == "gated":
            g = torch.sigmoid(self.gate(torch.cat([i, t], -1)))
            # Per-dimension convex mix: where g ~ 1 trust the image, where g ~ 0 trust the text.
            # The model learns, for each meme and each dimension, which modality to rely on. Shape [B,d].
            return g * i + (1 - g) * t
        # Attention fusion: treat [CLS, image, text] as a 3-token sequence and run self-attention.
        if f == "attention":
            # cls.expand(B,-1,-1) -> [B,1,d] (copy, no new memory); squeeze(1) -> [B,d]; stacking with i and t
            # on dim 1 gives tokens [B,3,d]; adding mod_emb [1,3,d] (broadcast over B) marks token identity.
            tokens = torch.stack([self.cls.expand(len(i), -1, -1).squeeze(1), i, t], 1) + self.mod_emb
            # Run the 2-layer encoder -> [B,3,d]; every token attends to the others, so the CLS output
            # mixes image and text information. Take the CLS position [:, 0] -> z [B,d].
            return self.encoder(tokens)[:, 0]

    # -----------------------------------------------------------------------------------------------
    # forward
    # Purpose : Full model call used in training and prediction.
    # Inputs  : img [B,768], txt [B,768].
    # Outputs : logits [B] (apply sigmoid to get P(hateful)).
    # Viva tip: Q: Why no sigmoid in forward? A: BCEWithLogitsLoss applies sigmoid internally in a
    #           numerically stable way (log-sum-exp trick); we apply sigmoid only at prediction time.
    # -----------------------------------------------------------------------------------------------
    def forward(self, img, txt):
        # fuse -> z [B,z_dim]; classifier -> [B,1]; squeeze(-1) -> [B] to match labels of shape [B].
        return self.classifier(self.fuse(img, txt)).squeeze(-1)
