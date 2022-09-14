
import torch, random, os, time, json, pdb
import datetime
import numpy as np
from models import build_model
from datasets import build_dataset, collate_fn
from utils import plot_logs
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Arguments(object):
    def __init__(self) -> None:
        print(f"This machine has {torch.cuda.device_count()} gpu...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rootpath = "/home/bebin.huang/Code/FoG_prediction/ECG_Object_Det/ECG_dataset" ## dataset path
        self.numfolds = 10
        self.seed = 10086
        self.batchsize = 128
        self.epochs = 200
        self.clip_max_norm = 0.15
        self.lr_drop = 80
        self.output_dir = "./outputs/"

def main():
    # pdb.set_trace()
    args = Arguments()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        logpath = os.path.join(args.output_dir, "log.txt")
        if os.path.exists(logpath):
            os.remove(logpath)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_chan, d_model, num_class, num_queries, aux_loss = 1, 128, 5, 10, True
    model, criterion, postprocessor = build_model(in_chan, d_model, num_class, num_queries, aux_loss=aux_loss)
    model.to(args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, 
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], 
        "lr": 1e-3}
        ]
    optimizer = torch.optim.Adam(param_dicts, lr=1e-3, weight_decay=1e-4)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)

    all_samples = [f for f in os.listdir(os.path.join(args.rootpath, "data")) if f.endswith(".txt")]
    random.shuffle(all_samples)
    # all_samples = all_samples[:2000]
    samples_each_fold = len(all_samples) // args.numfolds
    for fold in range(1):
        val_samples = all_samples[fold*samples_each_fold:(fold+1)*samples_each_fold]
        train_samples = [sam for sam in all_samples if sam not in val_samples]

        dataset_train = build_dataset(args.rootpath, train_samples)
        dataset_val = build_dataset(args.rootpath, val_samples)

        data_loader_train = DataLoader(dataset_train, args.batchsize, shuffle=True, collate_fn=collate_fn)
        data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)

        print("Start training...")
        start_time = time.time()
        # pdb.set_trace()
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, args.device, 
                epoch, args.clip_max_norm
            )
            lr_schedular.step()

            test_stats = evaluate(
                model, criterion, postprocessor, data_loader_val, args.device, args.output_dir
            )

            log_stats = {"epoch": epoch,
                        "n_params": n_parameters,
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        }

            if args.output_dir:
                if (epoch + 1) % args.lr_drop == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint_{epoch:04}.pth")
                    torch.save({
                        "epoch": epoch,
                        "args": args,
                        "model": model.state_dict(),
                        # "stats": log_stats,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_schedular.state_dict()
                    }, ckpt)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        plot_logs(args.output_dir, log_name="log.txt", fields=("loss", "loss_ce", "loss_bbox", "loss_giou", "class_error"))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Fold.{} Training time {}'.format(fold, total_time_str))

if __name__ == "__main__":
    main()