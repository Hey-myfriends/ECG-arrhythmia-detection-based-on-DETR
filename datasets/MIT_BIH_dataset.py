
import matplotlib.pyplot as plt
import numpy as np
import torch, wfdb, os, shutil, pdb, random
from torch.utils.data import Dataset, DataLoader

class MIT_BIH_dataset(Dataset):
    def __init__(self, root, samples) -> None:
        super().__init__()
        self.root = root
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # pdb.set_trace()
        data = np.loadtxt(os.path.join(self.root, "data/{}".format(self.samples[index])))
        gt_labels = np.loadtxt(os.path.join(self.root, "labels/{}".format(self.samples[index])))
        assert gt_labels.shape != torch.Size([0]), "gt_labels ndim is 0 (error)..."
        if gt_labels.ndim == 1:
            gt_labels = gt_labels.reshape(1, gt_labels.shape[0])
        # if gt_labels.ndim != 2:
        #     raise ValueError(f"gt_labels ndim error, shape is {gt_labels.shape}...")
        assert gt_labels.ndim == 2, f"gt_labels' dim must be 2, but now is {gt_labels.shape}..."

        data = torch.FloatTensor(data).unsqueeze(0)
        boxes = torch.FloatTensor(gt_labels[:, :2]) # (n_objects, 2)
        labels = torch.LongTensor(gt_labels[:, 2]) # (n_objects, )
        return data, boxes, labels

def collate_fn(batch: list):
    # pdb.set_trace()
    data, targets = [], []
    for b in batch:
        data.append(b[0])
        targets.append({"boxes": b[1], "labels": b[2]})

    # (1, L) -> (N, 1, L)
    data = torch.stack(data, dim=0)
    return data, targets

def Z_score(record: np.array):
    """
    record: [C, L]
    """
    mu, std = record.mean(axis=-1), record.std(axis=-1)
    return (record - mu[..., None]) / std[..., None]

def create_dataset(length=3, shift=2, fs=360):
    AAMI_MIT  = {'N': 'Nfe/jnBLR',# 0, 将19类信号分为五大类, 按照Advancement of Medical Instrumentation (AAMI)标准
             'S': 'SAJa', # 1
             'V': 'VEr', # 2
             'F': 'F', # 3
             'Q': 'Q?'} # 4
    ECG_R_list = np.array(['N', 'f', 'e', '/', 'j', 'n', 'B',
                           'L', 'R', 'S', 'A', 'J', 'a', 'V',
                           'E', 'r', 'F', 'Q', '?'])
    AAMI_MIT2 = {}
    labMap = {k: i for i, k in enumerate(AAMI_MIT.keys())}
    for k, v in AAMI_MIT.items():
        for s in v:
            AAMI_MIT2[s] = labMap[k]

    # pdb.set_trace()
    rootpath = "/home/bebin.huang/Code/FoG_prediction/ECG_Object_Det/mit-bih-arrhythmia-database-1.0.0/"
    person = [p[:3] for p in os.listdir(rootpath) if p.endswith(".dat")]
    outputPath = "/home/bebin.huang/Code/FoG_prediction/ECG_Object_Det/ECG_dataset/"
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath, ignore_errors=True)
    os.makedirs(outputPath+"data")
    os.makedirs(outputPath+"labels")

    counts = {v: 0 for v in labMap.values()}
    cnt = 0
    for candidates in person[:]:
        if not exists(rootpath+candidates+".hea", "MLII"):
            print(f"candidate {candidates} does not include MLII")
            continue
        print("Prepare data in candidate {}.....".format(candidates))
        annotations = wfdb.rdann(rootpath+candidates, "atr")
        records = wfdb.rdrecord(rootpath+candidates, physical=True, channel_names=["MLII"])
        records = records.p_signal.flatten()
        records = Z_score(records) ## Z-score
        index = np.isin(annotations.symbol, ECG_R_list)
        # pdb.set_trace()
        labels = np.array(annotations.symbol)[index]
        samples = annotations.sample[index]

        # time = np.arange(0, 30*fs) / fs
        # sampto = np.argwhere(samples < 30*fs)[-1, 0]
        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # ax.plot(time, records.p_signal.flatten()[:30*fs])
        # ax.scatter(time[samples[:sampto]], records.p_signal.flatten()[samples[:sampto]], marker="*", color="r")
        # ax.set(xlabel="Time(s)", ylabel="MLII")
        # plt.savefig(f"{candidates}.png", dpi=300)

        start, end = 0, length*fs
        while end < records.shape[0]:
            data = records[start:end]
            
            ## record gt labels
            lab_idx = (samples >= start)
            lab_idx *= (samples < end)
            cur_sam, cur_lab = samples[lab_idx], labels[lab_idx]
            gt_labels = []
            for i in range(lab_idx.sum()-1):
                x1, x2 = cur_sam[i]-start, cur_sam[i+1]-start
                x1 /= (length*fs)
                x2 /= (length*fs)
                gt_labels.append([(x1 + x2)/2, x2-x1, AAMI_MIT2[cur_lab[i]], int(candidates)]) # target labels are expected in format [center_x, w]
                counts[AAMI_MIT2[cur_lab[i]]] += 1
            # pdb.set_trace()
            start += shift * fs
            end += shift * fs
            if len(gt_labels) == 0: ## 片段中无目标，删除
                continue
            cnt += 1
            ## write to csv file
            np.savetxt(outputPath+f"data/{str(cnt).zfill(6)}.txt", data)
            np.savetxt(outputPath+f"labels/{str(cnt).zfill(6)}.txt", np.array(gt_labels))
            # print(f"No.{str(cnt).zfill(6)}: {len(gt_labels)} boxes")

    print(counts)

def exists(path, chan_name):
    assert os.path.exists(path), "path not exists"
    with open(path, "rb") as f:
        for line in f.readlines():
            if chan_name in line.decode().strip().split(" "):
                return True
    return False

if __name__ == "__main__":
    rootpath = "D:\\Desktop\\ECG分类研究\\dataset"
    # person = [p for p in os.listdir(rootpath) if p.endswith(".dat")]
    # print(person)

    # flag = exists(rootpath+"100.hea", "MLII")
    # print(flag)
    create_dataset()
    # all_files = [f for f in os.listdir(os.path.join(rootpath, "data")) if f.endswith(".txt")]
    # random.shuffle(all_files)
    # dataset = MIT_BIH_dataset(root=rootpath, samples=all_files[:1000])
    # pdb.set_trace()
    # print(dataset.__len__())
    # dataloader = DataLoader(dataset, 32, shuffle=True, collate_fn=collate_fn)
    # for data, gt_lab in dataloader:
    #     print(data.shape, gt_lab.keys())

