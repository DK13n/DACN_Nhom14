import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Model_arch.CDRes_ViTModel import CDRes_ViT
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import math, os, time
from collections import defaultdict
from src.Model_arch.Preprocess import make_train_dataset

class FASHead(nn.Module):
    def __init__(self, model,device, d_model, nums_class):
        super().__init__()
        self.backbone = model
        self.Dense = nn.Linear(d_model,nums_class)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)

        x = self.backbone(x)
        logits = self.Dense(x["fused"])

        return logits


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_acer(logits, labels):
    preds = torch.argmax(torch.softmax(logits,dim=1),dim=1)
    live_inx = labels==0
    spoof_inx = labels==1

    apcer = (preds[spoof_inx] == 0).float().mean().item() if spoof_inx.sum() > 0 else 0.0
    bpcer = (preds[live_inx] == 1).float().mean().item() if live_inx.sum() > 0 else 0.0

    acer = (apcer + bpcer) / 2
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}


@torch.no_grad()
def evulate(model, val, device, amp_dtype=None):

    model.eval()
    total = 0
    loss_sum = 0.0
    all_logits = []
    all_labels = []

    ce = nn.CrossEntropyLoss()

    for x, y in val:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.cuda.amp.autocast(
                enabled=False):
            logits, _ = model(x)
            loss = ce(logits, y)

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_acer(all_logits, all_labels)  # tính APCER/BPCER/ACER
    metrics["loss"] = loss_sum / max(total, 1)
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype=None, grad_clip=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    acc_sum  = 0.0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch["x"], batch["y"]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.cuda.amp.autocast(enabled=False):
            logits, _ = model(x)
            loss = ce(logits, y)

        if scaler:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        acc_sum  += accuracy_from_logits(logits, y) * bs

    return {
        "loss": loss_sum / max(total, 1),
        "acc":  acc_sum  / max(total, 1),
        "num":  total
    }


def fit(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-4,
    amp=True,                 # dùng mixed precision
    amp_dtype=torch.float16,  # hoặc torch.bfloat16 nếu GPU hỗ trợ
    grad_clip=None,
    scheduler_type="cosine",  # 'cosine' | 'step' | None
    step_size=5, gamma=0.1,   # cho step scheduler
    out_dir="./checkpoints",
    exp_name="fas_cdres_vit",
    early_stop_patience=5
):
    os.makedirs(out_dir, exist_ok=True)
    device = device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    scaler = GradScaler() if amp else None

    best_val = math.inf
    best_path = os.path.join(out_dir, f"{exp_name}_best.pt")
    patience = early_stop_patience
    history = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                        amp_dtype=amp_dtype if amp else None, grad_clip=grad_clip)
        val_metrics   = evulate(model, valid_loader, device, amp_dtype=amp_dtype if amp else None)

        if scheduler is not None:
            scheduler.step()

        # Lưu best theo val loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val},
                       best_path)
            patience = early_stop_patience  # reset
        else:
            patience -= 1

        dt = time.time() - t0
        log = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "val_loss":val_metrics['loss'],
            "APCER":val_metrics['APCER'],
            "BPCER":val_metrics['BPCER'],
            "ACER":val_metrics['ACER'],
            "time":       round(dt, 2),
            "early_stop_left": patience
        }
        history.append(log)
        print(f"[{epoch:03d}] lr={log['lr']:.2e} | "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"APCER={val_metrics['APCER']:.4f}, "
                f"BPCER={val_metrics['BPCER']:.4f}, "
                f"ACER={val_metrics['ACER']:.4f}")

        if patience <= 0:
            print("Early stopping!")
            break

    print(f"Best checkpoint saved at: {best_path} (val_loss={best_val:.4f})")
    return history, best_path


def train_model(
        model,
        device,
        csv_path,
        base_dir="",
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        amp=True,  # dùng mixed precision
        amp_dtype=torch.float16,  # hoặc torch.bfloat16 nếu GPU hỗ trợ
        grad_clip=None,
        scheduler_type="cosine",  # 'cosine' | 'step' | None
        step_size=5, gamma=0.1,  # cho step scheduler
        out_dir="./checkpoints",
        exp_name="fas_cdres_vit",
        early_stop_patience=5
):
    train_loader, val_loader = make_train_dataset(csv_path, base_dir)

    history, best_path = fit(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        amp=amp, amp_dtype=amp_dtype,
        scheduler_type=scheduler_type,
        out_dir=out_dir,
        exp_name=exp_name,
        early_stop_patience=early_stop_patience
    )

