import torch


class L1Loss(torch.nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, preds, gts):
            # 1. Isolate the SDF prediction (drops the 3 useless extra model outputs)
            if len(preds.shape) > 1 and preds.shape[-1] == 4:
                preds = preds[..., 0]
                
            # 2. Isolate the SDF ground truth (drops the XYZ coordinates)
            if len(gts.shape) > 1 and gts.shape[-1] == 4:
                gts = gts[..., 0]
                
            # 3. Final safety alignment (e.g., matching [4, 20000] exactly)
            if preds.shape != gts.shape:
                gts = gts.view_as(preds)
                
            # 4. Compute L1 loss, sum over points, mean over batch
            return self.loss(preds, gts).sum(-1).mean()
