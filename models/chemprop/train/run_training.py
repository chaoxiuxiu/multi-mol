from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model, build_pretrain_model, add_functional_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def run_training(args: Namespace, prompt: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)  # 获取 SMILES 字符串和目标数值，并将它们封装成一个 MoleculeDataset 对象。
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:  # 如果参数 args.separate_test_path 存在，则表示已经提供了单独的测试集路径，那么会使用 get_data 函数从单独的测试集路径加载测试集数据。
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path,
                            logger=logger)
    # 如果同时存在验证集和测试集路径，则将训练集设置为原始数据集。
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    # 如果存在验证集路径但不存在测试集路径，则使用 split_data 函数按照指定的拆分方式将数据集拆分成训练集和测试集。
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed, args=args, logger=logger)
    # 如果既不存在验证集路径也不存在测试集路径，则使用 split_data 函数按照指定的拆分方式将数据集拆分成训练集、验证集和测试集。
    else:
        print('=' * 100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                     seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)  # 调用函数 get_class_sizes(data) 来获取数据集中每个任务（属性）的类别大小。通常用于多类别分类任务。
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):#遍历 class_sizes 列表，i 是任务的索引，task_class_sizes 是该任务的类别大小分布。
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])
            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model from {args.checkpoint_path}')

            model = build_model(args, encoder_name=args.encoder_name)
            model.encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, encoder_name=args.encoder_name)

        if args.step == 'functional_prompt':
            add_functional_prompt(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in range(args.epochs):
            info(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                prompt=prompt,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                prompt=prompt,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            info(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

            test_preds = predict(
                model=model,
                prompt=prompt,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

                # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds = predict(
            model=model,
            prompt=prompt,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores


def pre_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    print(args)
    data = get_data(path=args.data_path, args=args, logger=logger)

    args.data_size = len(data)

    debug(f'Total size = {len(data)}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # try:
        #     writer = SummaryWriter(log_dir=save_dir)
        # except:
        #     writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model2 = model2.cuda()

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device
        criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        optimizer = Adam([{"params": model1.parameters()}, {"params": model2.parameters()}], lr=3e-5)
        # optimizer = Adam([{"params": model1.parameters()}], lr=3e-5)
        scheduler = ExponentialLR(optimizer, 0.99, -1)
        step_per_schedule = 500
        global_step = 0

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        loader = DataLoader(smiles,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=12,
                            drop_last=True)
        # Run training
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            debug = logger.debug if logger is not None else print
            model1.train()
            model2.train()
            # data.shuffle()
            num_iters = len(
                data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability
            iter_size = args.batch_size

            step = 'pretrain'
            for batch in tqdm(loader):
                # Run model
                emb1 = model1(step, False, batch,
                              None)  # def forward(self, step, prompt: bool, batch, features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
                emb2 = model2(step, True, batch, None)

                loss = criterion(emb1, emb2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                # logger.info(f'[{i}/{num_iters}] train loss {loss.item():.4f}')

                # save model
                if global_step % 1000 == 0:
                    snapshot(model1.encoder, global_step, dump_folder, 'original')
                    snapshot(model2.encoder, global_step, dump_folder, 'augment')
                if global_step % step_per_schedule == 0:
                    scheduler.step()
            logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')
    # return emb
