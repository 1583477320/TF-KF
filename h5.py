import h5py
import numpy as np

def convert_train_h5_to_bin(
    src_h5,
    dst_h5,
    test_subjects=('S9', 'S11')
):
    with h5py.File(src_h5, 'r') as f:
        part = f['part'][:]        # (N,17,2)
        S = f['S'][:]              # (N,17,3)
        center = f['center'][:]    # (N,2)
        scale = f['scale'][:]      # (N,)
        imgname = f['imgname'][:]  # bytes

    imgname = [n.decode('utf-8') for n in imgname]

    # ---------- 1. 构建 X / Y ----------
    X = (part - center[:, None, :]) / scale[:, None, None]
    Y = S - S[:, 0:1, :]   # root-relative

    # ---------- 2. train / test split ----------
    train_idx, test_idx = [], []
    for i, name in enumerate(imgname):
        if any(s in name for s in test_subjects):
            test_idx.append(i)
        else:
            train_idx.append(i)

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    names_tr = [imgname[i] for i in train_idx]
    names_te = [imgname[i] for i in test_idx]

    # ---------- 3. 写成 bin h5 ----------
    with h5py.File(dst_h5, 'w') as f:
        f.create_dataset('db_values_x_training', data=Xtr, dtype='float32')
        f.create_dataset('db_values_y_training', data=Ytr, dtype='float32')
        f.create_dataset('db_names_training',
                         data=np.array(names_tr, dtype='S'))

        f.create_dataset('db_values_x_test', data=Xte, dtype='float32')
        f.create_dataset('db_values_y_test', data=Yte, dtype='float32')
        f.create_dataset('db_names_test',
                         data=np.array(names_te, dtype='S'))

    print('Done. Train:', len(Xtr), 'Test:', len(Xte))

# convert_train_h5_to_bin(
#     src_h5='./data/h36m/annot/train.h5',
#     dst_h5='./data/h36m/full-2048.h5'
# )

import h5py

with h5py.File('./data/annot/train.h5', 'r') as f:
    def print_ds(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(print_ds)

with h5py.File('./data/annot/train.h5', 'r') as f:
    imgname = f['index']
    print(imgname[:])


    # import numpy as np
    # import matplotlib.pyplot as plt


    # # -----------------------------
    # # 你的数据（示例）
    # # -----------------------------
    # joints = np.array(imgname[:][1])


    # plt.scatter(joints[:,0], joints[:,1], c="r", s=30)
    # plt.axis("off")
    # plt.savefig("pose.png")
    # plt.show()

