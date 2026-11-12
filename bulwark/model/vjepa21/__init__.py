"""Meta's V-JEPA 2.1 ViT-B encoder code, vendored so that an installed package can build B1.

The files come from https://github.com/facebookresearch/vjepa2 at commit
204698b45b3712590f06245fbfba32d3be539812 (`bulwark.model.backbone.VJEPA_COMMIT`):
`vit_base` and `VisionTransformer` (app/vjepa_2_1/models/vision_transformer.py), the
blocks and attention they use (app/vjepa_2_1/models/utils/modules.py), the patch
embeddings (app/vjepa_2_1/models/utils/patch_embed.py), `trunc_normal_`
(src/utils/tensors.py) and `apply_masks` (src/masks/utils.py). They are trimmed to the
definitions the encoder needs, with imports pointing at this package and timm's
`drop_path` imported from `timm.layers`; the header of each file says what changed, and
the code of every definition kept is Meta's, unchanged (tests/test_model_vendored_vjepa.py
compares it with the pinned clone when one is present).

MIT License, Copyright (c) Meta Platforms, Inc. and affiliates: LICENSE in this folder,
which a B1 release ships as LICENSE-vjepa2 (`bulwark.inference.VJEPA_LICENSE`). See NOTICE.
"""
