import re
import argparse
import os
import time
from datetime import datetime
import logging
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.dataset import ModelNet10Dataset, SymsolDataset
from src.pascal_dataset import Pascal3D
from src.so3_utils import rotation_error, nearest_rotmat
from src.predictor import I2S


def create_dataloaders(args):
    if args.dataset_name.find('modelnet10') > -1:
        train_set = ModelNet10Dataset(args.dataset_path,
                                      train=True,
                                      limited=(args.dataset_name.find('limited') > -1))
        test_set = ModelNet10Dataset(args.dataset_path,
                                     train=False)
    elif args.dataset_name.find('pascal3d') > -1:
        train_set = Pascal3D(args.dataset_path,
                             train=True,
                             use_warp=args.dataset_name.find('warp') > -1,
                             use_synth=args.dataset_name.find('synth') > -1,
                            )
        test_set = Pascal3D(args.dataset_path,
                            train=False)
    elif args.dataset_name.find('symsol') > -1:
        train_set = SymsolDataset(args.dataset_path,
                                  train=True,
                                  set_number=args.dataset_name.count('I'),
                                  num_views=int(re.findall('\d+', args.dataset_name)[0]))
        test_set = SymsolDataset(args.dataset_path,
                                 train=False,
                                 set_number=args.dataset_name.count('I'),
                                 num_views=5000)
    else:
        raise TypeError('Invalid dataset name')


    print(f'{len(train_set)} train imgs; {len(test_set)} test imgs')

    args.img_shape = train_set.img_shape
    args.num_classes = train_set.num_classes
    args.class_names = train_set.class_names

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)
    return train_loader, test_loader, args


def evaluate_ll(args, model, test_loader):
    #log likelihood estimation
    model.eval()
    lls = []
    clss = []
    for batch_idx, batch in enumerate(test_loader):
        batch = {k:v.to(args.device) for k,v in batch.items()}
        probs = model.compute_probabilities(batch['img'], batch['cls'])

        gt_rotmats = batch['rot'].cpu()
        gt_inds = nearest_rotmat(gt_rotmats, model.eval_rotmats)
        gt_probs = probs[torch.arange(gt_rotmats.size(0)), gt_inds]
        log_likelihood = torch.log(gt_probs * model.eval_rotmats.shape[0] / np.pi**2)

        lls.append(log_likelihood.numpy())
        clss.append(batch['cls'].squeeze().cpu().numpy())

    lls = np.concatenate(lls)
    clss = np.concatenate(clss)

    per_class_ll = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_ll[args.class_names[i]] = lls[mask]

    np.save(os.path.join(args.fdir, f'eval_log_likelihood.npy'), per_class_ll)


def evaluate_error(args, model, test_loader):
    model.eval()
    errors = []
    clss = []
    for batch_idx, batch in enumerate(test_loader):
        batch = {k:v.to(args.device) for k,v in batch.items()}
        pred_rotmat = model.predict(batch['img'], batch['cls']).cpu()
        gt_rotmat = batch['rot'].cpu()
        err = rotation_error(pred_rotmat, gt_rotmat)
        errors.append(err.numpy())
        clss.append(batch['cls'].squeeze().cpu().numpy())

    errors = np.concatenate(errors)
    clss = np.concatenate(clss)

    per_class_err = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_err[args.class_names[i]] = errors[mask]

    np.save(os.path.join(args.fdir, 'eval.npy'), per_class_err)


def create_model(args):
    model = I2S(num_classes=args.num_classes,
                encoder=args.encoder,
                sphere_fdim=args.sphere_fdim,
                lmax=args.lmax,
                train_grid_rec_level=args.train_grid_rec_level,
                train_grid_n_points=args.train_grid_n_points,
                train_grid_include_gt=args.train_grid_include_gt,
                train_grid_mode=args.train_grid_mode,
                eval_grid_rec_level=args.eval_grid_rec_level,
                eval_use_gradient_ascent=args.eval_use_gradient_ascent,
                include_class_label=args.include_class_label,
               ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params: {num_params/1e6:.3f}M')

    model.train()
    return model


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    fname = f"{args.dataset_name}_{args.encoder.replace('_','-')}_seed{args.seed}"
    if args.include_class_label:
        fname += "_cls-label"

    if args.desc != '':
        fname += f"_{args.desc}"
    args.fdir = os.path.join(args.results_dir, fname)
    print(args.fdir)

    if not os.path.exists(args.fdir):
        os.makedirs(args.fdir)

    with open(os.path.join(args.fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers =  [logging.StreamHandler(),
                        logging.FileHandler(os.path.join(args.fdir, "log.txt"))]

    train_loader, test_loader, args = create_dataloaders(args)

    model = create_model(args)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr_initial,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay,
                                nesterov=bool(args.use_nesterov),
                               )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   args.lr_step_size,
                                                   args.lr_decay_rate)

    if os.path.exists(os.path.join(args.fdir, "checkpoint.pt")):
        # read the log to find the epoch
        checkpoint = torch.load(os.path.join(args.fdir, "checkpoint.pt"))
        if checkpoint['done']:
            exit()

        starting_epoch = checkpoint['epoch'] + 1
        epoch = starting_epoch
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        model.train()
    else:
        starting_epoch = 1

    data = []
    for epoch in range(starting_epoch, args.num_epochs+1):
        train_loss = 0
        train_acc = []
        time_before_epoch = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k:v.to(args.device) for k,v in batch.items()}
            loss, acc = model.compute_loss(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc.append(acc)

        train_loss /= batch_idx + 1
        train_acc_median = np.median(train_acc)

        test_loss = 0
        test_acc = []
        test_cls = []
        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = {k:v.to(args.device) for k,v in batch.items()}
            with torch.no_grad():
                loss, acc = model.compute_loss(**batch)

            test_loss += loss.item()
            test_acc.append(acc)
            test_cls.append(batch['cls'].squeeze(0).cpu().numpy())

        model.train()

        test_loss /= batch_idx + 1
        test_acc_median = np.median(test_acc)

        per_class_err = {}
        test_acc = np.array(test_acc).flatten()
        test_cls = np.array(test_cls).flatten()
        for i, cls in enumerate(args.class_names):
            per_class_err[cls] = f"{np.degrees(np.median(test_acc[test_cls == i])):.1f}"

        logger.info(str(per_class_err))

        data.append(dict(epoch=epoch,
                         time_elapsed=time.perf_counter() - time_before_epoch,
                         train_loss=train_loss,
                         test_loss=test_loss,
                         train_acc_median=train_acc_median,
                         test_acc_median=test_acc_median,
                         lr=optimizer.param_groups[0]['lr'],
                        ))
        lr_scheduler.step()

        # checkpointing
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'done': False,
                   }, os.path.join(args.fdir, "checkpoint.pt"))

        log_str = f"Epoch {epoch}/{args.num_epochs} | " \
                  + f"LOSS={train_loss:.4f}<{test_loss:.4f}> " \
                  + f"ROT ERR={np.degrees(test_acc_median):.2f}° | " \
                  + f"time={time.perf_counter() - time_before_epoch:.1f}s | " \
                  + f"lr={lr_scheduler.get_last_lr()[0]:.1e}"
        logger.info(log_str)
        time_before_epoch = time.perf_counter()


    if args.dataset_name.find('symsol') > -1:
        evaluate_ll(args, model, test_loader)
    else:
        # median rotation error
        evaluate_error(args, model, test_loader)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'done' : True,
               }, os.path.join(args.fdir, "checkpoint.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--encoder', type=str, default='resnet50_pretrained')

    parser.add_argument('--lmax', type=int, default=6,
                        help='Maximum degree of harmonics to use in spherical convolution')
    parser.add_argument('--sphere_fdim', type=int, default=512,
                        help='Feature dimension projected onto sphere')

    parser.add_argument('--train_grid_rec_level', type=int, default=3)
    parser.add_argument('--train_grid_n_points', type=int, default=4096)
    parser.add_argument('--train_grid_include_gt', type=int, default=0)
    parser.add_argument('--train_grid_mode', type=str, default='healpix',
                        choices=['healpix', 'random'])
    parser.add_argument('--eval_grid_rec_level', type=int, default=5)
    parser.add_argument('--eval_use_gradient_ascent', type=int, default=0)
    parser.add_argument('--include_class_label', type=int, default=0)

    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_initial', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--use_nesterov', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='modelnet10',
                        choices=['modelnet10',          # modelnet10 with 100 training views per instance
                                 'modelnet10-limited',  # modelnet10 with 20 training views per instance
                                 'pascal3d-warp-synth', # pascal3D with warping and synthetic data
                                 'symsolI-50000', # 5 classes of symsolI with 50k training views each
                                 'symsolII-50000', # symsol sphX with 50k training views each
                                 'symsolIII-50000', # symsol cylO with 50k training views each
                                 'symsolIIII-50000', # symsol tetX with 50k training views each
                                ]
                       )

    parser.add_argument('--num_workers', type=int, default=4,
                        help='workers used by dataloader')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
