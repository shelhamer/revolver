import os
import glob
import re
import setproctitle

import click
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from revolver.data import datasets, datatypes, prepare_loader
from revolver.model import models, prepare_model
from revolver.metrics import SegScorer


def evaluate(model, weights, dataset, datatype, split, count, shot, seed, gpu, hist_path, seg_path):
    print("evaluating {} with weights {} on {} {}-{}".format(model, weights, datatype, dataset, split))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device('cuda:0')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    prepare_data = datatypes[datatype]
    dataset = prepare_data(dataset, split, count=count, shot=shot)
    loader = prepare_loader(dataset, evaluation=True)

    model = prepare_model(model, dataset.num_classes, weights=weights).to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=dataset.ignore_index)

    total_loss = 0.
    metrics = SegScorer(len(dataset.classes))  # n.b. this is the full no. of classes, not the no. of model outputs
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, target, aux = data[:-2], data[-2], data[-1]
            inputs = [inp.to(device) if not isinstance(inp, list) else
                    [[i_.to(device) for i_ in in_] for in_ in inp] for inp in inputs]
            target = target.to(device)

            scores = model(*inputs)
            loss = loss_fn(scores, target)
            total_loss += loss.item()

            # segmentation evaluation
            _, seg = scores.data[0].max(0)
            metrics.update(seg.to('cpu').numpy(), target.to('cpu').numpy(), aux)
            # optionally save segs
            if seg_path is not None:
                seg = Image.fromarray(seg.to('cpu').numpy().astype(np.uint8), mode='P')
                save_id = f"{aux['slug']}_{aux.get('cls', 'all')}_{aux.get('inst', 'all')}"
                seg.save(f"{seg_path}/{save_id}.png")

    print("loss {}".format(total_loss / len(dataset)))
    for metric, score in metrics.score().items():
        score = np.nanmean(score)
        print("{:10s} {:.3f}".format(metric, score))

    if hist_path is not None:
        metrics.save(hist_path)


@click.command()
@click.argument('experiment', type=str)
@click.option('--model', type=click.Choice(models.keys()))
@click.option('--weights', type=click.Path())
@click.option('--dataset', type=click.Choice(datasets.keys()), default='voc')
@click.option('--datatype', type=click.Choice(datatypes.keys()), default='semantic')
@click.option('--split', type=str, default='valid')
@click.option('--count', type=int, default=None)
@click.option('--shot', type=int, default=1)
@click.option('--save_seg', is_flag=True, default=False)
@click.option('--seed', default=1337)
@click.option('--gpu', default=0)
def main(experiment, model, weights, dataset, datatype, split, count, shot, save_seg, seed, gpu):
    setproctitle.setproctitle("eval-{}".format(experiment))
    args = locals()
    print("args: {}".format(args))

    exp_dir = './experiments/{}/'.format(experiment)
    if not os.path.isdir(exp_dir):
        raise Exception("Experiment {} does not exist.".format(experiment))

    if weights:
        # evaluate given model
        evaluations = [weights]
    else:
        # evaluate all models in iteration order
        # but skip existing evaluations
        evaluations = sorted(glob.glob(exp_dir + '*.pth'))

    # template the output path
    count_ = 'dense' if count == -1 else "{}sparse".format(count) if count else 'randsparse'
    output_fmt = '-{}-{}-{}-{}-{}shot-{}'.format(dataset, datatype, split, count_, shot, seed)
    output_fmt = model + '-iter{}' + output_fmt

    for weights in evaluations:
        # make output path
        iter_ = re.search('iter(\d+).pth', weights).group(1)
        hist_path = exp_dir + 'hist-' + output_fmt.format(iter_)
        seg_path = None
        if save_seg:
            seg_path = exp_dir + output_fmt.format(iter_)
            os.makedirs(seg_path, exist_ok=True)
        # skip existing
        if os.path.isfile(hist_path + '.npz'):
            print("skipping existing {}".format(hist_path))
            continue
        evaluate(model, weights, dataset, datatype, split, count, shot, seed, gpu, hist_path, seg_path)

if __name__ == '__main__':
    main()
