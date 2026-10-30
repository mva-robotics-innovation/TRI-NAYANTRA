"""Train a brain-MRI classifier (transfer learning, ResNet-18) -> models/mri_classifier.pt
Dataset: any folder-per-class set, e.g. Kaggle "Brain Tumor MRI Dataset" (glioma / meningioma / pituitary / notumor):
   data/mri/Training/<class>/*.jpg   data/mri/Testing/<class>/*.jpg
Run:  pip install torch torchvision ; python train_mri.py --data data/mri --epochs 5
Research/education only. Validate on your own data before trusting any output."""
import argparse, torch, torchvision as tv
from torchvision import transforms as T
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/mri"); ap.add_argument("--epochs", type=int, default=5); a = ap.parse_args()
norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
tr = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.RandomRotation(10), T.RandomHorizontalFlip(), T.ToTensor(), norm])
te = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor(), norm])
dtr = tv.datasets.ImageFolder(f"{a.data}/Training", tr); dte = tv.datasets.ImageFolder(f"{a.data}/Testing", te)
ltr = torch.utils.data.DataLoader(dtr, 32, shuffle=True, num_workers=2); lte = torch.utils.data.DataLoader(dte, 64)
m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT); m.fc = torch.nn.Linear(512, len(dtr.classes))
dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4)
for ep in range(a.epochs):
    m.train()
    for x, y in ltr:
        opt.zero_grad(); torch.nn.functional.cross_entropy(m(x.to(dev)), y.to(dev)).backward(); opt.step()
    m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in lte: ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    print(f"epoch {ep+1}: test accuracy {ok/n:.3f}")
torch.save({"arch": "resnet18", "classes": dtr.classes, "state": m.cpu().state_dict()}, "models/mri_classifier.pt"); print("saved models/mri_classifier.pt")
