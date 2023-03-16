'''https://raw.githubusercontent.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions/master/Pascal3D/Pascal3D.py
'''
import glob
import scipy.io
import copy
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import skimage
import skimage.transform
from scipy.io import loadmat
import torch


def get_pascal_paths(folder, cate, mode):
    assert mode in ('train', 'val')
    img_paths = []
    annot_paths = []

    with open(os.path.join(folder, f'PASCAL/VOCdevkit/VOC2012/ImageSets/Main/{cate}_{mode}.txt'), 'r') as f:
        for l in f.readlines():
            num = l.split(' ')[0]
            if os.path.exists(os.path.join(folder, f'Images/{cate}_pascal/{num}.jpg')):
                img_paths.append(os.path.join(folder, f'Images/{cate}_pascal/{num}.jpg'))
                annot_paths.append(os.path.join(folder, f'Annotations/{cate}_pascal/{num}.mat'))

    return img_paths, annot_paths

def get_imagenet_paths(folder, cate, mode):
    assert mode in ('train', 'val')
    img_paths = []
    annot_paths = []
    with open(os.path.join(folder, f'Image_sets/{cate}_imagenet_{mode}.txt'), 'r') as f:
        for l in f.readlines():
            num = l.split('\n')[0]
            img_paths.append(os.path.join(folder, f'Images/{cate}_imagenet/{num}.JPEG'))
            annot_paths.append(os.path.join(folder, f'Annotations/{cate}_imagenet/{num}.mat'))

    return img_paths, annot_paths

def get_image_annotation_paths(directory, mode='train'):
    categories = ('aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
                  'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')

    assert mode in ('train', 'test')
    imgs = []
    rots = []
    cat_ids = []

    img_paths = []
    annot_paths = []
    for cate in categories:
        if mode == 'train':
            new_img_paths, new_annot_paths = get_pascal_paths(directory, cate, 'train')
            img_paths.extend(new_img_paths)
            annot_paths.extend(new_annot_paths)

            new_img_paths, new_annot_paths = get_imagenet_paths(directory, cate, 'train')
            img_paths.extend(new_img_paths)
            annot_paths.extend(new_annot_paths)

            new_img_paths, new_annot_paths = get_imagenet_paths(directory, cate, 'val')
            img_paths.extend(new_img_paths)
            annot_paths.extend(new_annot_paths)
        else:
            new_img_paths, new_annot_paths = get_pascal_paths(directory, cate, 'val')
            img_paths.extend(new_img_paths)
            annot_paths.extend(new_annot_paths)

    # filter out difficult or improperly annotated
    valid_img_paths = []
    valid_annot_paths = []
    for i, annot_path in enumerate(annot_paths):
        mat = loadmat(annot_path)
        obj = mat['record']['objects'][0][0][0][0]

        if obj['viewpoint'].dtype == np.float64:
            continue
        if 'distance' not in obj['viewpoint'].dtype.names:
            continue
        elif obj['viewpoint']['distance'][0][0][0][0] == 0:
            continue

        occluded = obj['occluded'][0][0]
        difficult = obj['difficult'][0][0]
        truncated = obj['truncated'][0][0]
        is_hard = occluded or difficult or truncated
        if is_hard:
            continue
        left, top, right, bottom = obj['bbox'][0].astype(np.uint16)
        valid_box = left < right and top < bottom
        if not valid_box:
            continue

        valid_img_paths.append(img_paths[i])
        valid_annot_paths.append(annot_paths[i])

    return valid_img_paths, valid_annot_paths


def get_camera_parameters(cam, principal_point, angle, image_size, distance):
    P = get_camera_matrix(cam, principal_point, angle, image_size, distance)
    intrinsic, extrinsic = split_camera_matrix(P)
    return intrinsic, extrinsic

def RQ(A):
    to_qr = A.transpose()[:, ::-1]
    Qhat, Rhat = np.linalg.qr(to_qr)
    return (Rhat[::-1, ::-1].transpose()), Qhat[:, ::-1].transpose()

def split_camera_matrix(P):
    extrinsic = np.zeros((4,4))
    intrinsic = np.zeros((3,3))
    extrinsic[3,3] = 1
    r,q  = RQ(P[:3,:3])
    for i in range(3):
        if r[i,i] < 0:
            q[i, :] *= -1
            r[:, i] *= -1

    assert(np.linalg.det(q) > 0)
    extrinsic[:3,:3] = q
    intrinsic = r
    intrinsic /= intrinsic[2,2]
    extrinsic[:3,3] = np.linalg.solve(intrinsic, P[:3, 3])

    if np.max((np.matmul(intrinsic, extrinsic[:3, :]) - P).reshape(-1)) > 0.1:
        print(P)
        print(np.matmul(intrinsic, extrinsic[:3, :]))
        raise Exception("Failed to split camera")
    return intrinsic, extrinsic


def get_camera_matrix(camera, principal_point, angle, image_size, distance):
    f = np.prod(camera)
    a = angle[0] *np.pi/180
    e = angle[1] *np.pi/180
    th = angle[2] *np.pi/180
    C = np.zeros(3)

    C[0] = distance*np.cos(e)*np.sin(a)
    C[1] = -distance*np.cos(e)*np.cos(a)
    C[2] = distance*np.sin(e)

    a = -a
    e = e-np.pi/2

    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0,np.sin(e), np.cos(e)]])
    R = np.matmul(Rx, Rz)

    R2d = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0,0,1]])
    intrinsic = np.array([[f, 0, 0], [0, f, 0], [0,0,-1]])
    intrinsic_w_rot = np.matmul(R2d, intrinsic)
    y_flip = np.array([[1, 0,0], [0,-1,0], [0,0,1]])
    intrinsic_w_flip = np.matmul(y_flip, intrinsic_w_rot)
    pp = principal_point # + np.array([image_size[1]/2.0, image_size[0]/2.0]) # image size is hxw
    trans_mat = np.array([[1.0, 0, pp[0]], [0, 1, pp[1]], [0,0,1]])
    intrinsic_w_trans = np.matmul(trans_mat, intrinsic_w_flip)

    extrinsic = np.zeros((4,4))
    extrinsic[:3,:3] = R
    extrinsic[3,3] = 1
    extrinsic[:3,3] = -np.matmul(R, C)
    P = np.matmul(intrinsic_w_trans, extrinsic[:3,:])
    return P

def get_back_proj_bbx(bbx, intrinsic):
    assert(len(bbx) == 4)
    minx = bbx[0]
    miny = bbx[1]
    maxx = bbx[2]
    maxy = bbx[3]
    points = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
    points = points.transpose()
    points_homo = np.ones((3,4))
    points_homo[:2, :] = points
    intrinsic_inv = np.linalg.inv(intrinsic)
    backproj = np.matmul(intrinsic_inv, points_homo)
    backproj /= np.linalg.norm(backproj, axis=0).reshape(1, -1)
    return backproj


def get_desired_camera(desired_imagesize, backproj, desired_up):
    z, radius_3d = get_minimum_covering_sphere(backproj.transpose())
    y = desired_up - np.dot(desired_up, z)*z
    y /= np.linalg.norm(y)
    x = -np.cross(y,z)
    R = np.stack([y,x,z], axis=0)# AXIS? TODO
    bp_reproj = np.matmul(R, backproj)
    bp_reproj/=bp_reproj[2,:].reshape(1, -1)
    f = 1/np.max(np.abs(bp_reproj[:2]).reshape(-1))

    intrinsic = np.array([[desired_imagesize*f/2, 0, desired_imagesize/2],
                          [0, desired_imagesize*f/2, desired_imagesize/2],
                          [0, 0, 1]])
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    return extrinsic, intrinsic


def get_minimum_covering_sphere(points):
    # points = nx3 array on unit sphere
    # returns point on unit sphere which minimizes the maximum distance to point in points
    # uses modified version of welzl
    points = np.copy(points)
    np.random.shuffle(points)
    def sphere_welzl(points, included_points, num_included_points):
        if len(points) == 0 or num_included_points == 3:
            return sphere_trivial(included_points[:num_included_points])
        else:
            p = points[0]
            rem = points[1:]
            cand_mid, cand_rad = sphere_welzl(rem, included_points, num_included_points)
            if np.linalg.norm(p-cand_mid) < cand_rad:
                return cand_mid, cand_rad
            included_points[num_included_points] = p
            return sphere_welzl(rem, included_points, num_included_points+1)
    buf = np.empty((3,3), dtype=np.float32)
    return sphere_welzl(points, buf, 0)


def sphere_trivial(points):
    if len(points) == 0:
        return np.array([1.0, 0,0]), 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        mid = (points[0] + points[1])/2
        diff = points-mid.reshape(1, -1)
        r = np.max(np.linalg.norm(diff, axis=1))
        return mid, r
    elif len(points) == 3:
        X = np.stack(points, axis=0)
        C = np.array([1,1,1])
        mid = np.linalg.solve(X, C)
        mid /= np.linalg.norm(mid)
        r = np.max(np.linalg.norm(points-mid.reshape(1, -1), axis=1))
        return mid, r
    raise Exception("2d welzl should not need 4 points")


class Pascal3DReal(Dataset):
    def __init__(self, directory='datasets', train=True, img_size=224, use_warp=True):
        self.warp = use_warp
        mode = 'train' if train else 'test'
        self.img_paths, self.annot_paths = get_image_annotation_paths(os.path.join(directory, 'PASCAL3D+_release1.1'),
                                                                      mode=mode)
        self.img_size = img_size

        self.num_classes = 12
        self.class_names = ('aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
                            'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annot_path = self.annot_paths[idx]

        with open(img_path, 'rb') as f:
            img_PIL = Image.open(f)
            img_PIL.convert('RGB')
            data = img_PIL.getdata()
            if isinstance(data[0], int) or len(data[0]) == img_PIL.size[1] * img_PIL.size[0]:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0]).reshape(img_PIL.size[1], img_PIL.size[0],1).repeat(3,2)
            else:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)


        obj = loadmat(annot_path)['record']['objects'][0][0][0][0]
        bbox = np.array(obj['bbox'][0], dtype=float)
        class_idx = obj['class'][0]

        viewpoint = obj['viewpoint']
        angle = [viewpoint['azimuth'][0][0][0][0],
                 viewpoint['elevation'][0][0][0][0],
                 viewpoint['theta'][0][0][0][0]
                ]
        cam = np.array([viewpoint['focal'][0][0][0][0],
                        viewpoint['viewport'][0][0][0][0]], dtype=float)

        principal_point = np.array([viewpoint['px'][0][0][0][0],
                                    viewpoint['px'][0][0][0][0]], dtype=float)
        distance = viewpoint['distance'][0][0][0][0]

        if self.warp:
            flip = np.random.randint(2)
            if flip==1:
                angle *= np.array([-1.0, 1.0, -1.0])
                img_full = img_full[:, ::-1, :]
                bbox[0] = img_full.shape[1] - bbox[0]
                bbox[2] = img_full.shape[1] - bbox[2]
                principal_point[0] = img_full.shape[1] - principal_point[0]
            # # change up direction of warp
            desired_up = np.random.normal(0,0.4,size=(3))+np.array([3.0, 0.0, 0.0])
            desired_up[2] = 0
            desired_up /= np.linalg.norm(desired_up)
            # # jitter bounding box
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox[0::2] += np.random.uniform(-bbox_w*0.1, bbox_w*0.1, size=(2))
            bbox[1::2] += np.random.uniform(-bbox_h*0.1, bbox_h*0.1, size=(2))
        else:
            desired_up = np.array([1.0, 0.0, 0.0])

        intrinsic, extrinsic = get_camera_parameters(cam, principal_point, angle, img_full.shape, distance)
        back_proj_bbx = get_back_proj_bbx(bbox, intrinsic)

        extrinsic_desired_change, intrinsic_new = get_desired_camera(self.img_size, back_proj_bbx, desired_up)
        extrinsic_after = np.matmul(extrinsic_desired_change, extrinsic)

        P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
        P /= P[2,2]
        Pinv = np.linalg.inv(P)
        transform = skimage.transform.ProjectiveTransform(Pinv)
        im = img_full.astype(np.float32)/255
        warped_image = skimage.transform.warp(im, transform, output_shape=(self.img_size, self.img_size), mode='constant', cval=0.0)

        rot = torch.from_numpy(extrinsic_after[:3,:3]).to(torch.float32)

        ret = dict(
            img=torch.from_numpy(warped_image).permute(2,0,1).to(torch.float32),
            cls=torch.tensor((self.class_names.index(class_idx),), dtype=torch.long),
            rot=rot,
        )

        return ret

    @property
    def img_shape(self):
        return (3, self.img_size, self.img_size)


class Pascal3DSynth(Dataset):
    def __init__(self, directory='datasets', img_size=224):
        self.img_size = img_size

        self.num_classes = 12
        self.class_names = ('aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
                            'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')
        self.class_ids = ('02691156', '02834778', '02858304', '02876657', '02924116', '02958343',
                          '03001627', '04379243', '03790512', '04256520', '04468005', '03211117')

        self.files = []
        for num in self.class_ids:
            self.files.extend(glob.glob(os.path.join(directory, f'syn_images_cropped_bkg_overlaid/{num}/*/*.png')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        class_idx = self.class_ids.index(fname.split('/')[-3])

        img_full = Image.open(fname)
        img_full = np.array(img_full.getdata()).reshape(img_full.size[1], img_full.size[0],3).astype(np.float32) / 255

        base_name = os.path.split(fname)[1]
        a, e, t = [int(x[1:]) for x in base_name.split('_')[-4:-1]]
        distance = 4.0
        bbox = [0, 0, img_full.shape[1], img_full.shape[0]]
        cam = 3000
        principal_point = np.array([img_full.shape[1]/2, img_full.shape[0]/2], dtype=np.float32)

        flip = np.random.randint(2)
        if flip:
            a = -a
            t = -t
            img_full = img_full[:, ::-1, :]
            bbox[0] = img_full.shape[1] - bbox[0]
            bbox[2] = img_full.shape[1] - bbox[2]
            principal_point[0] = img_full.shape[1] - principal_point[0]

        desired_up = np.array([3.0, 0.0, 0.0]) + np.random.normal(0,0.4,size=(3))
        desired_up[2] = 0
        desired_up /= np.linalg.norm(desired_up)
        # # jitter bounding box
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox[0::2] += np.random.uniform(-bbox_w*0.1, bbox_w*0.1, size=(2))
        bbox[1::2] += np.random.uniform(-bbox_h*0.1, bbox_h*0.1, size=(2))

        angle = np.array([a,e,-t])
        intrinsic, extrinsic = get_camera_parameters(cam, principal_point, angle, img_full.shape, distance)
        back_proj_bbx = get_back_proj_bbx(bbox, intrinsic)
        extrinsic_desired_change, intrinsic_new = get_desired_camera(self.img_size, back_proj_bbx, desired_up)
        extrinsic_after = np.matmul(extrinsic_desired_change, extrinsic)

        P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
        P /= P[2,2]
        Pinv = np.linalg.inv(P)
        transform = skimage.transform.ProjectiveTransform(Pinv)

        warped_image = skimage.transform.warp(img_full, transform, output_shape=(self.img_size, self.img_size), mode='constant', cval=0.0)

        rot = torch.from_numpy(extrinsic_after[:3,:3]).to(torch.float32)

        ret = dict(
            img=torch.from_numpy(warped_image).permute(2,0,1).to(torch.float32),
            cls=torch.tensor((class_idx,), dtype=torch.long),
            rot=rot,
        )

        return ret

    @property
    def img_shape(self):
        return (3, self.img_size, self.img_size)

class Pascal3D(Dataset):
    def __init__(self, datasets_dir, train:bool, use_warp: bool=False, use_synth: bool=False):
        if not train:
            assert use_warp == False and use_synth == False

        self.real_dataset = Pascal3DReal(datasets_dir, train=train, use_warp=use_warp)

        self.use_synth = use_synth
        if use_synth:
            self.synth_dataset = Pascal3DSynth(datasets_dir)

        self.img_shape = self.real_dataset.img_shape
        self.num_classes = self.real_dataset.num_classes
        self.class_names = self.real_dataset.class_names

    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        else:
            # TODO: implement it better so every minibatch has same mix
            idx = np.random.randint(len(self.synth_dataset))
            return self.synth_dataset[idx]

    def __len__(self):
        if self.use_synth:
            return 4 * len(self.real_dataset)
        else:
            return len(self.real_dataset)

if __name__ == "__main__":
    dataset = Pascal3DReal(train=False, use_warp=False)
    a = np.arange(len(dataset))
    np.random.shuffle(a)
    a = a[:10]

    imgs = []
    rots = []
    clss = []
    for i in a:
        b = dataset[i]
        imgs.append((b['img'].permute(1, 2, 0) * 255).numpy().astype(np.uint8))
        rots.append(b['rot'].numpy())
        clss.append(b['cls'].numpy())

    imgs = np.array(imgs)
    rots = np.array(rots)
    clss = np.array(clss)
    np.savez('mini_pascal.npz', imgs=imgs, rots=rots, clss=clss)
