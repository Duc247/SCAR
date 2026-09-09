"""Independent M3-DPF architecture; M0-M3 implementations remain unchanged."""
from training.models.cmspa_net import CMSPANet
from training.models.modules.dpf import DualPathologyFusion


class M3DPF(CMSPANet):
    def __init__(self, config=None, **kwargs):
        if (kwargs.get("ablation") or "M3").upper() != "M3":
            raise ValueError("M3-DPF requires the M3 encoder/SSPANet backbone")
        super().__init__(config, **kwargs)
        if self.ablation != "M3" or self.config.n_skip < 2:
            raise ValueError("M3-DPF requires M3 and at least two skips")
        if self.num_classes != 4:
            raise ValueError("M3-DPF requires four canonical classes")
        self.config.architecture = "m3_dpf"
        self.config.dpf_bottleneck_width = self.config.get("dpf_bottleneck_width", 64)
        self.config.dpf_skip_width = self.config.get("dpf_skip_width", 32)
        for key in ("dpf_bottleneck_width", "dpf_skip_width"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.config.get("dpf_loss_version", 1) != 1:
            raise ValueError("Unsupported M3-DPF loss version")
        self.config.dpf_loss_version = 1
        width = int(64 * self.config.resnet.width_factor)
        self.cross_fusion = DualPathologyFusion(16 * width, self.config.fused_channels,
                                               self.config.dpf_bottleneck_width)
        self.feature_fusion[1] = DualPathologyFusion(4 * width, 4 * width,
                                                   self.config.dpf_skip_width, auxiliary=True)

    def forward(self, cine, psir, t2w, return_aux=False):
        self._validate_inputs(cine, psir, t2w)
        images = [x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x for x in (cine, psir, t2w)]
        encoded = [encoder(x) for encoder, x in zip(
            (self.transformer1, self.transformer2, self.transformer3), images)]
        fused = self.cross_fusion(*[attention(pair[0]) for attention, pair in zip(
            (self.sspanet_cine, self.sspanet_psir, self.sspanet_t2w), encoded)])
        skips, auxiliary = [], None
        for i, fusion in enumerate(self.feature_fusion):
            values = [pair[1][i] for pair in encoded]
            if i == 1 and return_aux:
                skip, auxiliary = fusion(*values, return_aux=True)
            else:
                skip = fusion(*values)
            skips.append(skip)
        logits = self.segmentation_head(self.decoder(fused, skips))
        return {"logits": logits, "aux_logits": auxiliary} if return_aux else logits
