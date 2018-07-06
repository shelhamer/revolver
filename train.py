import logging
import os
import setproctitle
import shutil
import subprocess
import sys

import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multiprocessing
from torch.autograd import Variable

from revolver.data import datasets, datatypes, prepare_loader
from revolver.model import models, prepare_model
from revolver.model.loss import CrossEntropyLoss2D
from revolver.metrics import SegScorer

from evaluate import evaluate


def pevaluate(q):
    # keep evaluating from the queue until done as signalled by None
    while True:
        args = q.get()
        if args is None:
            q.task_done()
            break
        evaluate(*args)
        q.task_done()


@click.command()
@click.argument('experiment', type=str)
@click.option('--model', type=click.Choice(models.keys()))
@click.option('--weights', type=str, default=None)
@click.option('--dataset', type=click.Choice(datasets.keys()), default='sbdd')
@click.option('--datatype', type=click.Choice(datatypes.keys()), default='semantic')
@click.option('--split', type=str, default='train')
@click.option('--val_dataset', type=click.Choice(datasets.keys()), default='sbdd')
@click.option('--val_split', type=str, default='val')
@click.option('--class_group', type=click.Choice(['all', '0', '1', '2', '3']), default='all')
@click.option('--count', type=int, default=None)  # -1 -> dense, None -> random in [0, 100], >= 1 -> count
@click.option('--shot', type=int, default=1)
@click.option('--lr', default=1e-5)
@click.option('--max_iter', type=int, default=int(1e5))
@click.option('--seed', default=1337)
@click.option('--gpu', default=0)
@click.option('--do-eval/--no-eval', default=True)
def main(experiment, model, weights, dataset, datatype, split, val_dataset, val_split, class_group, count, shot, lr, max_iter, seed, gpu, do_eval):
    setproctitle.setproctitle(experiment)
    version = subprocess.check_output(['git', 'describe', '--always'], universal_newlines=True).strip()
    # experiment metadata
    args = locals()

    exp_dir = './experiments/{}/'.format(experiment)
    if os.path.isdir(exp_dir):
        click.confirm(click.style("{} already exists. Do you want to "
            "obliterate it and continue?".format(experiment), fg='red'),
            abort=True)
        shutil.rmtree(exp_dir)
    try:
        os.makedirs(exp_dir, exist_ok=True)
    except:
        raise Exception("Could not create experiment dir {}".format(exp_dir))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    logging.basicConfig(filename=exp_dir + 'log', level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("training %s", experiment)
    logging.info("args: %s", args)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # spawn persistent evaluation process
    if do_eval:
        mp_ctx = multiprocessing.get_context('spawn')
        q = mp_ctx.JoinableQueue()
        p = mp_ctx.Process(target=pevaluate, args=(q,))
        p.start()

    # filter classes by group for heldout experiments
    classes_to_filter = None
    if class_group != 'all':
        if dataset == 'davis':
            raise Exception("Class groups don't exist for DAVIS video data.")
        class_group = int(class_group)
        # divide classes into quarters and take background + the given quarter
        group_size = len(datasets[dataset].classes) // 4
        group_idx = 1 + class_group * group_size
        group_classes = range(group_idx, group_idx + group_size)
        classes_to_filter = list(set(range(1, len(datasets[dataset].classes))) - set(group_classes))

    dataset_name = dataset
    prepare_data = datatypes[datatype]
    dataset = prepare_data(dataset_name, split, classes_to_filter, count, shot)
    loader = prepare_loader(dataset)
    val_dataset_name = val_dataset or dataset_name

    model_name = model
    model = prepare_model(model, dataset.num_classes, weights).cuda()

    loss_fn = CrossEntropyLoss2D(size_average=True, ignore_index=dataset.ignore_index)
    learned_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.SGD(learned_params, lr=lr, momentum=0.99, weight_decay=0.0005)

    iter_order = int(np.log10(max_iter) + 1 )  # for pretty printing

    epoch = 0
    iteration = 0
    losses = []
    model.train()
    while iteration < max_iter:
        logging.info("epoch %d", epoch)
        epoch += 1
        train_loss = 0.
        for i, data in enumerate(loader):
            inputs, target, aux = data[:-2], data[-2], data[-1]
            model.set_way(aux.get('num_cl', dataset.num_classes))
            inputs = [Variable(inp).cuda() if not isinstance(inp, list) else
                    [[Variable(i_).cuda() for i_ in in_] for in_ in inp] for inp in inputs]
            target = Variable(target).cuda(async=True)

            scores = model(*inputs)
            loss = loss_fn(scores, target)
            loss.backward()

            train_loss += loss.data[0]
            losses.append(loss.data[0])
            if iteration % 20 == 0:
                logging.info("%s", "iter {iteration:{iter_order}d} loss {mean_loss:02.5f}".format(iteration=iteration, iter_order=iter_order, mean_loss=np.mean(losses)))
                losses = []

            if iteration % 4000 == 0:
                # snapshot
                logging.info("snapshotting...")
                snapshot_path = exp_dir + 'snapshot-iter{iteration:0{iter_order}d}.pth'.format(iteration=iteration, iter_order=iter_order)
                torch.save(model.state_dict(), snapshot_path)
                # evaluate
                if do_eval:
                    logging.info("evaluating...")
                    hist_path = exp_dir + 'hist-iter{iteration:0{iter_order}d}'.format(iteration=iteration, iter_order=iter_order)
                    try:
                        # wait for the last evalution if it's still running
                        q.join()
                    except:
                        pass
                    # carry out evaluation in independent process for determinism and speed
                    q.put((model_name, snapshot_path, val_dataset_name, datatype, val_split, count, shot, False, seed, gpu, hist_path))

            # update
            opt.step()
            opt.zero_grad()
            iteration += 1
        logging.info("%s", "train loss = {:02.5f}".format(train_loss / len(dataset)))

    # signal to evaluation process that training is done
    if do_eval:
        q.put(None)

if __name__ == '__main__':
    main()
