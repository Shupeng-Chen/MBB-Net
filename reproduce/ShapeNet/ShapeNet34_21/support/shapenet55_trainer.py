import os
import torch
from tqdm import tqdm
from reproduce.ShapeNet.ShapeNet34_21.support.loss import ShapeNetLoss
from reproduce.ShapeNet.ShapeNet34_21.support.metrics import calc_shapenet_metrics, AverageMeter, calc_acc


class ShapeNet55Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.cfg = cfg

        # 接入新的 C++ Loss 模块
        self.criterion = ShapeNetLoss(alpha=cfg.get('ALPHA', 0.4), cd_scale=1.0)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg['GAMMA'])

        self.ckpt_dir = os.path.join('checkpoints/ShapeNet', cfg['DATASET_NAME'])
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_cd = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        cd_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for partial, gt, label in pbar:
            partial, gt, label = partial.cuda(), gt.cuda(), label.cuda()

            self.optimizer.zero_grad()
            out_list, logits = self.model(partial)

            # 这里的 criterion 内部现在全都是 C++ 算子了
            loss, loss_comp, loss_ce = self.criterion(out_list, logits, gt, label)

            loss.backward()
            self.optimizer.step()

            acc = calc_acc(logits, label)

            loss_meter.update(loss.item(), partial.size(0))
            cd_meter.update(loss_comp.item(), partial.size(0))
            acc_meter.update(acc, partial.size(0))

            pbar.set_postfix({
                'Loss': f"{loss_meter.avg:.4f}",
                'CD': f"{cd_meter.avg:.4f}",
                'Acc': f"{acc_meter.avg * 100:.2f}%"
            })

        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        metrics = {'simple': AverageMeter(), 'moderate': AverageMeter(), 'hard': AverageMeter()}
        f1_metrics = {'simple': AverageMeter(), 'moderate': AverageMeter(), 'hard': AverageMeter()}
        acc_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Epoch {epoch} Eval")
            for res_dict, gt, label in pbar:
                gt, label = gt.cuda(), label.cuda()

                moderate_partial = res_dict['moderate'].cuda()
                _, logits = self.model(moderate_partial)
                acc = calc_acc(logits, label)
                acc_meter.update(acc, label.size(0))

                for difficulty in ['simple', 'moderate', 'hard']:
                    partial = res_dict[difficulty].cuda()
                    out_list, _ = self.model(partial)

                    final_pred = out_list[-1]

                    # 使用封装好的 C++ 极速测评函数
                    cd_val, f1_val = calc_shapenet_metrics(final_pred, gt)

                    metrics[difficulty].update(cd_val, partial.size(0))
                    f1_metrics[difficulty].update(f1_val, partial.size(0))

        avg_cd = (metrics['simple'].avg + metrics['moderate'].avg + metrics['hard'].avg) / 3.0
        avg_f1 = (f1_metrics['simple'].avg + f1_metrics['moderate'].avg + f1_metrics['hard'].avg) / 3.0

        print(f"\n--- Evaluation Results (Epoch {epoch}) ---")
        print(
            f"CD(L2*1000) -> Simple: {metrics['simple'].avg:.3f} | Mod: {metrics['moderate'].avg:.3f} | Hard: {metrics['hard'].avg:.3f} | Avg: {avg_cd:.3f}")
        print(
            f"F1-Score    -> Simple: {f1_metrics['simple'].avg:.3f} | Mod: {f1_metrics['moderate'].avg:.3f} | Hard: {f1_metrics['hard'].avg:.3f} | Avg: {avg_f1:.3f}")
        print(f"Accuracy    -> {acc_meter.avg * 100:.2f}%")

        if avg_cd < self.best_cd:
            self.best_cd = avg_cd
            save_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"[*] New Best CD! Saved to {save_path}")