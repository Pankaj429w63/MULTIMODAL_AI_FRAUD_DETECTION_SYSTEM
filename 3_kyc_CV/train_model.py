import torch
import torchvision.models as models

model = models.swin_t(weights="IMAGENET1K_V1")
model.head = torch.nn.Linear(model.head.in_features, 1)

torch.save(model.state_dict(), "kyc_model.pt")
print("KYC Swin Transformer model saved (baseline)")
