import os
import subprocess
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import lmdb
import cv2
from tqdm import tqdm


def generate_numpy_dataset(file_name):
    class_names = ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
                   'night_stand', 'sofa', 'table', 'toilet')

    viewID2quat = pickle.load(open(f'ModelNet10-SO3/{file_name}.Rawjpg.lmdb/viewID2quat.pkl', 'rb'), encoding='latin1')

    L = len(viewID2quat)
    data = dict(
        imgs = np.zeros((L, 224, 224, 1), dtype=np.uint8),
        rots = np.zeros((L, 3, 3), dtype=np.float32),
        cat_ids = np.zeros((L), dtype=int),
    )

    lmdb_env = lmdb.open(f'ModelNet10-SO3/{file_name}.Rawjpg.lmdb')

    for i, (view_name, quat) in tqdm(enumerate(viewID2quat.items()), desc=file_name, total=L):
        # need to put quat in scalar-last to use scipy rot package
        a,b,c,d = quat
        data['rots'][i] = R.from_quat((b,c,d,a)).as_matrix()

        data['cat_ids'][i] = class_names.index('_'.join(view_name.split('_')[:-1]))

        with lmdb_env.begin() as txn:
            raw_data = txn.get(view_name.encode('utf8'))
            npy_arr = np.frombuffer(raw_data, np.uint8)
            img = cv2.imdecode(npy_arr, cv2.IMREAD_COLOR)
        assert np.allclose(img[...,0], img[...,1])
        data['imgs'][i] = img[...,[0]]

    return data

if not os.path.exists('ModelNet10-SO3'):
    # downlaod and unpack it
    subprocess.run(["wget", "http://isis-data.science.uva.nl/shuai/datasets/ModelNet10-SO3.tar.gz"])
    subprocess.run(["tar", "xzvf", "ModelNet10-SO3.tar.gz"])
    subprocess.run(["rm", "ModelNet10-SO3.tar.gz"])

if not os.path.exists('modelnet10'):
    os.mkdir('modelnet10')

np.savez_compressed('modelnet10/modelnet10_train', **generate_numpy_dataset('train_100V'))
np.savez_compressed('modelnet10/modelnet10_limited_train', **generate_numpy_dataset('train_20V'))
np.savez_compressed('modelnet10/modelnet10_test', **generate_numpy_dataset('test_20V'))
