from model_runner.model_provider import get_model
import os
import h5py
import time
from random import randint
from itertools import cycle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from nets.inception_resnet_v2 import InceptionResNetV2
from helper.data_lodaer import build_inception_test_transform,build_inception_train_transform

def load_file(fl):
    """
    Loads text files for joints and mid-layer features.
    """
    if len(fl) > 10:
        # It might be a tuple or a path string depending on caller
        # Original code logic handles tuples (y, x)
        pass 

    # Handling tuple input (y_file, x_file)
    if isinstance(fl, tuple):
        yfl = fl[0]
        xfl = fl[1]
        
        with open(yfl, "r") as f: # Changed rb to r as it's text
            data = f.read().strip().split(' ')
            y_d = [np.float32(val) for val in data if val] # filter empty strings
            y_d = np.asarray(y_d, dtype=np.float32) / 1000.0
            
        with open(xfl, "r") as f:
            data = f.read().strip().split(' ')
            x_d = [np.float32(val) for val in data if val]
            x_d = np.asarray(x_d, dtype=np.float32)
        return (y_d, x_d)
    else:
        # Single file loading (assuming joints or similar)
        with open(fl, "r") as f:
            data = f.read().strip().split(' ')
            y_d = [np.float32(val) for val in data if val]
            y_d = np.asarray(y_d, dtype=np.float32) / 1000.0
        return y_d

def load_file_nodiv(fl):
    with open(fl, "r") as f:
        data = f.read().strip().split(' ')
        y_d = [np.float32(val) for val in data if val]
        y_d = np.asarray(y_d, dtype=np.float32)
    return y_d

def prepare_db(params, is_training):
    base_file = params['data_dir_y'] + "/joints"
    est_file = params['data_dir_x'] + "/fl_" + str(params['n_input'])
    max_count = params['max_count']
    print("Dataset loading from:  %s, %s " % (params['data_dir_x'], params['data_dir_y']))
    
    if is_training:
        lst_act = params["train_lst_act"]
    else:
        lst_act = params["test_lst_act"]
        
    db_names = []
    seq_id_names = []
    start = time.time()
    acto_cnt = 0
    seq_id = 0
    seq_y = []
    seq_x = []
    
    for actor in lst_act:
        tmp_folder = base_file + '/' + actor + "/"
        if not os.path.exists(tmp_folder):
            continue
        lst_sq = os.listdir(tmp_folder)
        acto_cnt += 1
        
        for sq in lst_sq:
            end = time.time()
            passed_time = end - start
            cnt = lst_sq.index(sq) + 1
            print("%s, (%i/%i)-%s loading... %i loaded, time: %s " % (actor, cnt, len(lst_sq), sq, len(db_names), passed_time))
            
            joint_tmp_folder = base_file + '/' + actor + "/" + sq + "/"
            mid_tmp_folder = est_file + '/' + actor + "/" + sq + "/"

            if not os.path.exists(joint_tmp_folder):
                continue
            if not os.path.exists(mid_tmp_folder):
                continue

            joint_id_list = os.listdir(joint_tmp_folder)
            mid_id_list = os.listdir(mid_tmp_folder)

            common_lst = [id for id in joint_id_list if id in mid_id_list]
            common_id_lst = sorted([int(f[0:-4]) for f in common_lst if f.endswith('.txt')])

            joint_list = [base_file + '/' + actor + '/' + sq + '/' + str(p1) + ".txt" for p1 in common_id_lst]
            midlayer_list = [est_file + '/' + actor + '/' + sq + '/' + str(p1) + ".txt" for p1 in common_id_lst]

            f_list = list(zip(joint_list, midlayer_list))

            pool = ThreadPool(8) # Limit threads to prevent system overload
            results = pool.map(load_file, f_list)
            pool.close()

            for r in results:
                seq_y.append(np.hstack((seq_id, r[0])))
                seq_x.append(np.hstack((seq_id, r[1])))
            db_names.extend(f_list)

            seq_id_names.append(str(seq_id) + "|" + actor + "|" + sq)
            seq_id += 1
            if len(db_names) > max_count:
                return (np.asarray(seq_x, dtype=np.float32), np.asarray(seq_y, dtype=np.float32), db_names, seq_id_names)

    return (np.asarray(seq_x, dtype=np.float32), np.asarray(seq_y, dtype=np.float32), db_names, seq_id_names)

def prepare_prediction_db(params, is_training):
    base_file = params['data_dir_y'] + "/joints"
    max_count = params['max_count']
    
    if is_training:
        lst_act = params["train_lst_act"]
    else:
        lst_act = params["test_lst_act"]
        
    db_names = []
    seq_id_names = []
    start = time.time()
    acto_cnt = 0
    seq_id = 0
    seq_y = []
    seq_x = []
    
    for actor in lst_act:
        tmp_folder = base_file + '/' + actor + "/"
        if not os.path.exists(tmp_folder):
            continue
        lst_sq = os.listdir(tmp_folder)
        acto_cnt += 1
        
        for sq in lst_sq:
            end = time.time()
            passed_time = end - start
            cnt = lst_sq.index(sq) + 1
            print("%s, (%i/%i)-%s loading... %i loaded, time: %s " % (actor, cnt, len(lst_sq), sq, len(db_names), passed_time))
            
            joint_tmp_folder = base_file + '/' + actor + "/" + sq + "/"

            if not os.path.exists(joint_tmp_folder):
                continue

            joint_id_list = os.listdir(joint_tmp_folder)
            common_id_lst = sorted([int(f[0:-4]) for f in joint_id_list if f.endswith('.txt')])

            joint_list = [base_file + '/' + actor + '/' + sq + '/' + str(p1) + ".txt" for p1 in common_id_lst]

            pool = ThreadPool(8)
            results = pool.map(load_file, joint_list)
            pool.close()
            
            for r in results:
                # For prediction DB, we only have Y (target), X is implicit or shifted
                # Original code appends (seq_id, results[r])
                seq_y.append(np.hstack((seq_id, r)))
            
            db_names.extend(joint_list)
            seq_id_names.append(str(seq_id) + "|" + actor + "|" + sq)
            seq_id += 1
            if len(db_names) > max_count:
                return (np.asarray(seq_x, dtype=np.float32), np.asarray(seq_y, dtype=np.float32), db_names, seq_id_names)

    return (np.asarray(seq_x, dtype=np.float32), np.asarray(seq_y, dtype=np.float32), db_names, seq_id_names)

def load_and_save_db(params):
    if params['is_forcasting'] == 1:
        (db_values_x_training, db_values_y_training, db_names_training, seq_id_names_training) = prepare_prediction_db(params, is_training=True)
        (db_values_x_test, db_values_y_test, db_names_test, seq_id_names_test) = prepare_prediction_db(params, is_training=False)
    else:
        (db_values_x_training, db_values_y_training, db_names_training, seq_id_names_training) = prepare_db(params, is_training=True)
        (db_values_x_test, db_values_y_test, db_names_test, seq_id_names_test) = prepare_db(params, is_training=False)
        
    fl = params["data_bin"] + "-" + str(len(db_values_x_training)) + "-" + str(len(db_values_x_test)) + ".h5"
    
    # Ensure directory exists
    db_dir = os.path.dirname(fl)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    h5f = h5py.File(fl, 'w')
    h5f.create_dataset('db_values_x_training', data=db_values_x_training)
    h5f.create_dataset('db_values_y_training', data=db_values_y_training)
    h5f.create_dataset('db_names_training', data=[s.encode('utf-8') for s in db_names_training]) # Store strings as bytes
    h5f.create_dataset('seq_id_names_training', data=[s.encode('utf-8') for s in seq_id_names_training])
    h5f.create_dataset('db_values_x_test', data=db_values_x_test)
    h5f.create_dataset('db_values_y_test', data=db_values_y_test)
    h5f.create_dataset('db_names_test', data=[s.encode('utf-8') for s in db_names_test])
    h5f.create_dataset('seq_id_names_test', data=[s.encode('utf-8') for s in seq_id_names_test])
    h5f.close()
    return (db_values_x_training, db_values_y_training, db_names_training, db_values_x_test, db_values_y_test, db_names_test)

def load_from_bin_db(params):
    fl = params["data_bin"]
    with h5py.File(fl, 'r') as f:
        db_values_x_training = f['db_values_x_training'][()].astype(dtype=np.float32)
        db_values_y_training = f['db_values_y_training'][()].astype(dtype=np.float32)
        db_names_training = [s.decode('utf-8') for s in f['db_names_training'][()]] # Decode bytes to str
        db_values_x_test = f['db_values_x_test'][()].astype(dtype=np.float32)
        db_values_y_test = f['db_values_y_test'][()].astype(dtype=np.float32)
        db_names_test = [s.decode('utf-8') for s in f['db_names_test'][()]]
        
    return (db_values_x_training, db_values_y_training, db_names_training, db_values_x_test, db_values_y_test, db_names_test)

import numpy as np

import numpy as np

def prepare_sequences(params, db_values_x, db_values_y, db_names):
    p_count = params['seq_length']
    # 如果 params 中没有 max_count，或者设为 None/负数，则视为加载全部
    max_count = params.get('max_count', None)
    
    # 结果列表
    X_D, Y_D, F_L, S_L, R_L = [], [], [], [], []
    G_L = [] # 保持原样，虽然未被使用
    
    # 当前正在构建的序列缓冲区
    Y_d, X_d, F_l, r_l = [], [], [], []
    
    # 初始化 ID，防止第一条数据 ID 为 0 时逻辑判断出错
    prev_sq_id = -1 
    curr_id = 0
    
    # 辅助函数：用于判断是否已经达到了 max_count 限制
    def is_reached_limit(current_len, limit):
        if limit is None or limit < 0:
            return False
        return current_len >= limit

    # 辅助函数：处理缓冲区（填充并保存）
    def flush_buffer(sequence_id):
        nonlocal Y_d, X_d, F_l, r_l
        
        if len(Y_d) == 0:
            return

        # --- 填充逻辑 ---
        residual = p_count - (len(Y_d) % p_count)
        # 只有当余数不为0（即不满一个 p_count）时才填充
        if residual < p_count:
            # 填充特征和标签
            Y_d.extend([Y_d[-1]] * residual)
            X_d.extend([X_d[-1]] * residual)
            # 填充文件名（如果 F_l 不为空）
            last_f = F_l[-1] if len(F_l) > 0 else None
            F_l.extend([last_f] * residual)
            # 填充掩码（0代表填充）
            r_l.extend([0] * residual)
            
        # --- 保存逻辑 ---
        # 经过填充后，长度一定是 p_count (或者原本就是 p_count)
        if len(Y_d) == p_count:
            S_L.append(sequence_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
        
        # 清空缓冲区
        Y_d, X_d, F_l, r_l = [], [], [], []

    for item_id in range(db_values_x.shape[0]):
        sq_id = int(db_values_x[item_id][0])
        x = db_values_x[item_id][1:]
        y = db_values_y[item_id][1:]
        
        # 修复逻辑：获取文件名
        f = db_names[item_id] if db_names is not None else None

        # 检测序列 ID 是否变化
        if prev_sq_id != sq_id:
            # 如果有旧的序列数据未处理，先处理（填充并保存）旧序列
            flush_buffer(curr_id)
            
            # 更新 ID 状态
            prev_sq_id = sq_id
            curr_id = sq_id

        # 向当前缓冲区添加数据
        Y_d.append(y)
        X_d.append(x)
        F_l.append(f)
        r_l.append(1) # 1 代表真实数据
        
        # 检查当前缓冲区是否已满达到序列长度
        if len(Y_d) == p_count and p_count > 0:
            # 保存已满的序列（不进行填充，因为正好满）
            S_L.append(curr_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
            # 清空缓冲区
            Y_d, X_d, F_l, r_l = [], [], [], []
            
            # 检查是否达到调试限制
            if is_reached_limit(len(Y_D), max_count):
                return (np.asarray(X_D, dtype=np.float32), np.asarray(Y_D, dtype=np.float32), 
                        np.asarray(F_L), np.asarray(G_L, dtype=np.float32), np.asarray(S_L, dtype=np.float32), np.asarray(R_L, dtype=np.int32))

    # --- 处理最后剩余的数据 ---
    # 循环结束后，如果缓冲区还有数据（长度不足 p_count 的尾部数据），进行填充保存
    flush_buffer(curr_id)

    return (np.asarray(X_D, dtype=np.float32), np.asarray(Y_D, dtype=np.float32), 
            np.asarray(F_L), np.asarray(G_L, dtype=np.float32), np.asarray(S_L, dtype=np.float32), np.asarray(R_L, dtype=np.int32))


def prepare_sequences_next_frame(params, db_values_x, db_values_y, db_names):
    """
    构造"预测下一帧"的序列数据
    
    输入: 第 t 帧的特征
    输出: 第 t+1 帧的 3D 姿态
    
    X: [f1, f2, ..., f49]  (前 seq_length-1 帧)
    Y: [y2, y3, ..., y50]  (后 seq_length-1 帧，即下一帧)
    """
    p_count = params['seq_length']
    max_count = params.get('max_count', None)
    
    X_D, Y_D, F_L, S_L, R_L = [], [], [], [], []
    G_L = []
    
    Y_d, X_d, F_l, r_l = [], [], [], []
    
    prev_sq_id = -1 
    curr_id = 0
    
    def is_reached_limit(current_len, limit):
        if limit is None or limit < 0:
            return False
        return current_len >= limit

    def flush_buffer(sequence_id):
        nonlocal Y_d, X_d, F_l, r_l
        
        if len(Y_d) <= 1:
            return

        residual = p_count - (len(Y_d) % p_count)
        if residual < p_count:
            Y_d.extend([Y_d[-1]] * residual)
            X_d.extend([X_d[-1]] * residual)
            last_f = F_l[-1] if len(F_l) > 0 else None
            F_l.extend([last_f] * residual)
            r_l.extend([0] * residual)
            
        if len(Y_d) == p_count:
            S_L.append(sequence_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
        
        Y_d, X_d, F_l, r_l = [], [], [], []

    for item_id in range(db_values_x.shape[0]):
        sq_id = int(db_values_x[item_id][0])
        x = db_values_x[item_id][1:]
        y = db_values_y[item_id][1:]
        
        f = db_names[item_id] if db_names is not None else None

        if prev_sq_id != sq_id:
            flush_buffer(curr_id)
            prev_sq_id = sq_id
            curr_id = sq_id

        Y_d.append(y)
        X_d.append(x)
        F_l.append(f)
        r_l.append(1)
        
        if len(Y_d) == p_count and p_count > 0:
            S_L.append(curr_id)
            Y_D.append(Y_d)
            X_D.append(X_d)
            F_L.append(F_l)
            R_L.append(r_l)
            Y_d, X_d, F_l, r_l = [], [], [], []
            
            if is_reached_limit(len(Y_D), max_count):
                return (np.asarray(X_D, dtype=np.float32), np.asarray(Y_D, dtype=np.float32), 
                        np.asarray(F_L), np.asarray(G_L, dtype=np.float32), np.asarray(S_L, dtype=np.float32), np.asarray(R_L, dtype=np.int32))

    flush_buffer(curr_id)

    X_arr = np.asarray(X_D, dtype=np.float32)
    Y_arr = np.asarray(Y_D, dtype=np.float32)
    
    X_input = X_arr[:, :-1, :]
    Y_target = Y_arr[:, 1:, :]
    F_arr = np.asarray(F_L)
    F_input = F_arr[:, :-1] if F_arr.dtype == object or len(F_arr.shape) > 1 else F_arr
    R_L_arr = np.asarray(R_L, dtype=np.int32)
    R_input = R_L_arr[:, :-1]
    
    return (X_input, Y_target, F_input, np.asarray(G_L, dtype=np.float32), 
            np.asarray(S_L, dtype=np.float32), R_input)

def prepare_prediction_sequences(params, db_values_y, db_names, seq_rel):
    fsc = params['forcast_sequence_count']
    seed_length = params['seed_length']
    forcast_length = params['forcast_length']
    seq_id_lst = seq_rel["seq_idx_lst"]
    forcast_id_lst = []
    
    # random selection
    np.random.shuffle(seq_id_lst)
    selected_lst = seq_id_lst[:fsc]
    X_Seed = []
    Y_Forcast = []
    DB_names_Forcast = []
    
    for s in selected_lst:
        id_lst = seq_rel[s]
        seed_start = -1
        for seq_ed in id_lst:
            s_idx = np.where(db_values_y[:, 0] == seq_ed)[0]
            sequence = db_values_y[s_idx]
            names_sequence = db_names[s_idx]
            
            if seed_start < 0:
                full_length = sequence.shape[0]
                tmp_len = full_length - (seed_length + forcast_length)
                if tmp_len > 0:
                    seed_start = randint(0, tmp_len)
                    seed_end = seed_start + seed_length
                    sequence_end = seed_start + seed_length + forcast_length
                    
                    x = sequence[seed_start:seed_end, 1:]
                    y = sequence[seed_start:sequence_end, 1:]
                    f = names_sequence[seed_start:sequence_end, 1:]
                    fd = [0] * (seed_end - seed_start)
                    fd.extend([1] * (sequence_end - seed_end))
                    X_Seed.append(x)
                    Y_Forcast.append(y)
                    forcast_id_lst.append(fd)
                    DB_names_Forcast.append(f)
                    
    return np.asarray(X_Seed), np.asarray(Y_Forcast), np.asarray(DB_names_Forcast), np.asarray(forcast_id_lst)

from helper import utils as ut


def prepare_training_set(
    params,
):
    train_seqs, test_seqs = scan_h36m_dataset_by_subject(params["h36m_root"])

    X_train_raw, Y_train_raw, names_train = \
        build_or_load_observations(
            train_seqs,
            params["cnn_model"],
            params["device"],
            cache_dir=params.get("cache_dir", "cache_obs")
        )

    X_test_raw, Y_test_raw, names_test = \
        build_or_load_observations(
            test_seqs,
            params["cnn_model"],
            params["device"],
            cache_dir=params.get("cache_dir", "cache_obs")
        )

    if params.get("subsample", 1) > 1:
        print("Subsampling...")
        X_train_raw, Y_train_raw, names_train, seq_rel_train = \
            subsample_sequences_h36m(params, X_train_raw, Y_train_raw, names_train, params["subsample"])

        X_test_raw, Y_test_raw, names_test, seq_rel_test = \
            subsample_sequences_h36m(params, X_test_raw, Y_test_raw, names_test, params["subsample"])
 
    norm_mode = params.get("normalise_data", 0)

    if norm_mode in [1, 3]:
        X_train_raw, X_test_raw, mean, std = \
            ut.normalise_data(X_train_raw, X_test_raw)
        params["x_mean"] = mean
        params["x_std"]  = std

    if norm_mode in [2, 3]:
        Y_train_raw, Y_test_raw, mean, std = \
            ut.normalise_data(Y_train_raw, Y_test_raw)
        params["y_mean"] = mean
        params["y_std"]  = std

    predict_next_frame = params.get("predict_next_frame", False)
    
    if predict_next_frame:
        print("Using 'predict next frame' mode...")
        X_train, Y_train, F_train, G_train, S_train, R_L_train = \
            prepare_sequences_next_frame(
                params,
                X_train_raw,
                Y_train_raw,
                names_train
            )

        X_test, Y_test, F_test, G_test, S_test, R_L_test = \
            prepare_sequences_next_frame(
                params,
                X_test_raw,
                Y_test_raw,
                names_test
            )
    else:
        X_train, Y_train, F_train, G_train, S_train, R_L_train = \
            prepare_sequences(
                params,
                X_train_raw,
                Y_train_raw,
                names_train
            )

        X_test, Y_test, F_test, G_test, S_test, R_L_test = \
            prepare_sequences(
                params,
                X_test_raw,
                Y_test_raw,
                names_test
            )

    return (
        params,
        X_train, Y_train, F_train, G_train, S_train, R_L_train,
        X_test,  Y_test,  F_test,  G_test,  S_test,  R_L_test
    )

import os
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def build_or_load_observations(
    h36m_sequences,          # List[dict]
    cnn_model,
    device,
    cache_dir="cache_obs",
    batch_size=64,
):
    """
    每个 sequence:
        {
            "name": "S1_Walking_1",
            "frames": List[str],        # 图像路径
            "gt": np.ndarray (T, gt_dim)
        }
    """
    
    model = InceptionResNetV2(num_classes=51)
    model.load_state_dict(torch.load(cnn_model, map_location=device))

    os.makedirs(cache_dir, exist_ok=True)
    model.eval().to(device)

    seq = h36m_sequences
    name = seq["name"]
    frames = seq["frames"]
    cache_path = os.path.join(cache_dir, f"{name}.npz")

    # -----------------------
    # 1. 从缓存加载
    # -----------------------
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X = data["X"]
        Y = data["Y"]
        print(f"[Cache] Loaded {name}")

    # -----------------------
    # 2. 重新跑 CNN 推理
    # -----------------------
    else:
        frames = seq["frames"]
        Y_feat = seq["gt"]
        T = len(frames)

        X_list = []

        for i in tqdm(range(0, T, batch_size), desc=name):
            batch_frames = frames[i:i+batch_size]

            imgs = load_and_preprocess_images(batch_frames, build_inception_train_transform())  
            imgs = imgs.to(device)

            preds,_ = model(imgs)        # (B, obs_dim)
            preds = preds.cpu().numpy()

            X_list.append(preds)
        # batch_frames = frames[0:4]

        # imgs = load_and_preprocess_images(batch_frames, build_inception_train_transform())  
        # imgs = imgs.to(device)

        # preds, _ = model(imgs)        # (B, obs_dim)
        # preds = preds.cpu().numpy()
    
        # X_list.append(preds)

        X_feat = np.concatenate(X_list, axis=0)

        # -----------------------
        # 添加 idx 列
        # -----------------------
        idx_col = seq["seq_ids"].reshape(-1, 1).astype(np.float32)

        X = np.concatenate([idx_col, X_feat], axis=1)   # (T, 52)
        Y = np.concatenate([idx_col, Y_feat], axis=1)   # (T, 52)

        assert len(X) == len(Y), f"Length mismatch in {name}"

        np.savez_compressed(cache_path, X=X, Y=Y)
        print(f"[Cache] Saved {name}")

    return X, Y, frames

import torch
from PIL import Image


def load_and_preprocess_images(frame_paths, transform):
    """
    输入:
        frame_paths : List[str or Path]
        transform   : torchvision transform

    输出:
        images : Tensor[B, C, H, W]
    """

    imgs = []

    for p in frame_paths:
        img = Image.open(p).convert("RGB")

        if transform is not None:
            img = transform(img)     # Tensor[C,H,W]

        imgs.append(img)

    # stack 成 batch
    images = torch.stack(imgs, dim=0)   # (B,C,H,W)
    return images


import h5py
from pathlib import Path

def scan_h36m_dataset_by_subject(data_root):
    """
    txt 中图像名:
        S1_Directions_1.54138969_000076.jpg

    实际路径:
        data/images/S1/Directions_1.54138969_000076.jpg

    在 scan 阶段完成:
        - mm -> m
        - flatten
        - torch tensor
        - build_seq_ids（sequence_id）

    返回:
        train_data, valid_data
        其中每个 dict 包含:
            frames   : List[str]
            gt       : Tensor (N, 51)
            seq_ids  : np.ndarray (N,)   # ⭐ 新增
    """

    from pathlib import Path
    import h5py
    import numpy as np
    import torch

    data_root = Path(data_root)
    img_root  = data_root / "images"
    annot_dir = data_root / "annot"

    # -----------------------------
    # 工具函数：构造真实图像路径
    # -----------------------------
    def build_image_path(img_name: str, split: str):
        # subject  = img_name.split("_", 1)[0]     # S1
        # filename = img_name.split("_", 1)[1]     # Directions_1.54138969_000076.jpg
        if split == "train":
            return str(img_root / split / img_name)
        else:
            return str(img_root / "test" / img_name)

    # -----------------------------
    # 工具函数：extract sequence key
    # -----------------------------
    def extract_seq_key_from_name(img_name: str):
        """
        S1_Directions_1.54138969_000076.jpg
        -> S1_Directions_1_54138969
        """
        stem = img_name.replace(".jpg", "")
        prefix, _ = stem.rsplit("_", 1)
        # subject, rest = prefix.split("_", 1)
        # action, cam_take = rest.split("_")
        # cam_id, take_id = cam_take.split(".")
        return f"{prefix}"

    # -----------------------------
    # build_seq_ids（帧级）
    # -----------------------------
    def build_seq_ids_from_img_names(img_names):
        seq_map = {}
        seq_ids = []
        next_id = 0

        for name in img_names:
            key = extract_seq_key_from_name(name)
            if key not in seq_map:
                seq_map[key] = next_id
                next_id += 1
            seq_ids.append(seq_map[key])

        return np.asarray(seq_ids, dtype=np.int32), seq_map

    # -----------------------------
    # 加载一个 split
    # -----------------------------
    def load_split(split):
        img_txt = annot_dir / f"{split}_images.txt"
        h5_path = annot_dir / f"{split}.h5"

        # ---------- load image list ----------
        with open(img_txt) as f:
            img_names = [l.strip() for l in f if l.strip()]

        frames = [build_image_path(n,split) for n in img_names]

        # ---------- build sequence ids ----------
        seq_ids, seq_map = build_seq_ids_from_img_names(img_names)

        # ---------- load pose ----------
        with h5py.File(h5_path, "r") as h5_file:
            S = h5_file["S"][:]      # (N,17,3)

        assert len(frames) == len(S) == len(seq_ids), \
            f"{split}: length mismatch"

        # ---------- pose preprocess ----------
        gt = []
        for pose3d in S:
            pose3d = pose3d.astype(np.float32) / 1000.0   # mm -> m
            pose3d = torch.from_numpy(pose3d).view(-1)    # (51,)
            gt.append(pose3d)

        gt = torch.stack(gt)   # (N,51)

        return {
            "name": split,
            "frames": frames,          # List[str]
            "gt": gt,                  # Tensor (N,51)
            "seq_ids": seq_ids,        # ⭐ 新增 (N,)
            "seq_map": seq_map         # 可选，调试用
        }

    train_data = load_split("train")
    valid_data = load_split("valid")

    print(f"Train frames: {len(train_data['frames'])}")
    print(f"Train sequences: {len(train_data['seq_map'])}")
    print(f"Valid frames: {len(valid_data['frames'])}")
    print(f"Valid sequences: {len(valid_data['seq_map'])}")

    return train_data, valid_data


def get_seq_indexes(params, S_L):
    # S_L is list of sequence IDs
    # Logic seems to be flattening the sequences into a batchable format
    # Here we just return an index list and the sequence list
    index_list = np.arange(len(S_L))
    return index_list, S_L

import torch
import numpy as np

def prepare_batch_tensors(z_raw, target_raw, mask_raw, device):
    """
    将原始 numpy 数据转换为模型需要的 Tensor 格式
    z_raw: (Batch, Seq, NOUT) - 观测数据
    target_raw: (Batch, Seq, NOUT) - 标签/真值
    mask_raw: (Batch, Seq) - 掩码 (1表示有效数据，0表示填充)
    """
    z_tensor = torch.from_numpy(z_raw).float().to(device)
    target_tensor = torch.from_numpy(target_raw).float().to(device)
    mask_tensor = torch.from_numpy(mask_raw).float().to(device)
    
    return z_tensor, target_tensor, mask_tensor


import numpy as np


def subsample_sequences_h36m(
    X_list,
    Y_list,
    names,
    subsample
):
    """
    等价于旧版 subsample_frames，但适用于 sequence list 结构。
    
    subsample = k
    实际步长 = k + 1
    """

    if subsample <= 0:
        return X_list, Y_list, names, None

    step = subsample + 1

    new_X = []
    new_Y = []
    new_names = []
    seq_rel = {}      # old_seq_id -> [new_seq_ids]

    new_id = 0

    for seq_id, (X, Y, name) in enumerate(zip(X_list, Y_list, names)):
        T = len(X)

        # 交错拆分
        for offset in range(step):
            idxs = list(range(offset, T, step))
            if len(idxs) < 2:
                continue

            X_sub = X[idxs]
            Y_sub = Y[idxs]

            new_X.append(X_sub)
            new_Y.append(Y_sub)
            new_names.append(f"{name}_sub{offset}")

            # seq_rel 映射
            if seq_id not in seq_rel:
                seq_rel[seq_id] = [new_id]
            else:
                seq_rel[seq_id].append(new_id)

            new_id += 1

    print(f"Subsample: {len(X_list)} → {len(new_X)} sequences")
    return new_X, new_Y, new_names, seq_rel
