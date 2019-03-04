import os
import glob
import re
import setproctitle

import click
import numpy as np

import torch
from torch.autograd import Variable

from revolver.data import datasets, datatypes, prepare_loader
from revolver.model import models, prepare_model
from revolver.model.loss import CrossEntropyLoss2D
from revolver.metrics import SegScorer


def evaluate(model, weights, dataset, datatype, split, count, shot, multi, seed, gpu, output):
    print("evaluating {} with weights {} on {} {}-{}".format(model, weights, datatype, dataset, split))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    prepare_data = datatypes[datatype]
    dataset = prepare_data(dataset, split, count=count, shot=shot, multi=multi)
    loader = prepare_loader(dataset, evaluation=True)

    model = prepare_model(model, dataset.num_classes, weights=weights).cuda()
    model.eval()

    loss_fn = CrossEntropyLoss2D(size_average=True, ignore_index=dataset.ignore_index)

    total_loss = 0.
    metrics = SegScorer(len(dataset.classes))  # n.b. this is the full no. of classes, not the no. of model outputs
    for i, data in enumerate(loader):
        inputs, target, aux = data[:-2], data[-2], data[-1]
        inputs = [Variable(inp, volatile=True).cuda() if not isinstance(inp, list) else
                [[Variable(i_, volatile=True).cuda() for i_ in in_] for in_ in inp] for inp in inputs]
        target = Variable(target, volatile=True).cuda(async=True)

        scores = model(*inputs)
        loss = loss_fn(scores, target)
        total_loss += loss.data[0]

        # segmentation evaluation
        _, seg = scores.data[0].max(0)
        metrics.update(seg.cpu().numpy(), target.data.cpu().numpy(), aux)

    print("loss {}".format(total_loss / len(dataset)))
    for metric, score in metrics.score().items():
        score = np.nanmean(score)
        print("{:10s} {:.3f}".format(metric, score))

    if output:
        metrics.save(output)


@click.command()
@click.argument('experiment', type=str)
@click.option('--model', type=click.Choice(models.keys()))
@click.option('--weights', type=click.Path())
@click.option('--dataset', type=click.Choice(datasets.keys()), default='voc')
@click.option('--datatype', type=click.Choice(datatypes.keys()), default='semantic')
@click.option('--split', type=str, default='valid')
@click.option('--count', type=int, default=None)
@click.option('--shot', type=int, default=1)
@click.option('--multi', is_flag=True, default=False)
@click.option('--seed', default=1337)
@click.option('--gpu', default=0)
def main(experiment, model, weights, dataset, datatype, split, count, shot, multi, seed, gpu):
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
    multi_ = '-multi' if multi else ''
    output_fmt = '-{}-{}-{}-{}-{}shot-{}{}'.format(dataset, datatype, split, count_, shot, seed, multi_)
    output_fmt = exp_dir + 'hist-' + model + '-iter{}' + output_fmt

    for weights in evaluations:
        # make output path
        iter_ = re.search('iter(\d+).pth', weights).group(1)
        output = output_fmt.format(iter_)
        # skip existing
        if os.path.isfile(output + '.npz'):
            print("skipping existing {}".format(output))
            continue
        evaluate(model, weights, dataset, datatype, split, count, shot, multi, seed, gpu, output)

if __name__ == '__main__':
    main()
