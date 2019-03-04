import os

import click
import numpy as np

import torch
from torch.autograd import Variable

from revolver.data import datasets, datatypes, prepare_loader
from revolver.model import models, prepare_model
from revolver.model.loss import CrossEntropyLoss2D
from revolver.metrics import SegScorer


@click.command()
@click.argument('exp')
@click.option('--model', type=click.Choice(models.keys()))
@click.option('--weight_iter', type=str)
@click.option('--dataset', type=click.Choice(datasets.keys()), default='voc')
@click.option('--datatype', type=click.Choice(datatypes.keys()), default='semantic')
@click.option('--split', type=str, default='valid')
@click.option('--seed', default=1337)
@click.option('--gpu', default=0)
def main(exp, model, weight_iter, dataset, datatype, split, seed, gpu):
    print("evaluating {} with weights {} on {}".format(model, exp, dataset))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    weights = f"experiments/{exp}/snapshot-iter{weight_iter}.pth"
    output_dir = f"experiments/{exp}"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    results = []
    data_name = dataset
    model_name = model
    for count in [2,4,8,16,32,64,128]:
        print(f"{count} annotations")
        prepare_data = datatypes[datatype]
        dataset = prepare_data(data_name, split, count=count)
        loader = prepare_loader(dataset, shuffle=False)

        model = prepare_model(model_name, dataset.num_classes).cuda()
        model.load_state_dict(torch.load(weights))
        model.eval()

        loss_fn = CrossEntropyLoss2D(size_average=True, ignore_index=dataset.ignore_index)

        total_loss = 0.
        metrics = SegScorer(len(dataset.classes))  # n.b. this is the full no. of classes, not the no. of model outputs
        for i, data in enumerate(loader):
            inputs, target, aux = data[:-2], data[-2], data[-1]
            inputs = [Variable(in_, volatile=True).cuda() for in_ in inputs]
            target = Variable(target, volatile=True).cuda(async=True)

            scores = model(*inputs)
            loss = loss_fn(scores, target)
            total_loss += loss.data[0]

            # segmentation evaluation
            _, seg = scores.data[0].max(0)
            metrics.update(seg.cpu().numpy(), target.data.cpu().numpy(), aux)

        print("loss {}".format(total_loss / len(dataset)))
        for metric, score in metrics.score().items():
            if score.size > 1:
                score = np.nanmean(score)
            print("{:10s} {:.3f}".format(metric, score))
        np.save(f"{output_dir}/hist_iter{weight_iter}_{count}_annos.npy", metrics.hist)


if __name__ == '__main__':
    main()
