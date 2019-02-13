import numpy as np
import click

from transducer.metrics import SegScorer

# trained nets are evaluated on all the validation data
# for semantic segmentation, we compute the mean positive IU for each split on the heldout classes
# and averages across splits to compute the final accuracy

@click.command()
@click.argument('exp_name')
@click.argument('histname')
def main(exp_name, histname):

    result_dir = f'./experiments/{exp_name}'

    def make_histname(it, histname):
        prefix, suffix  = histname.split('iter')
        suffix = f'iter{it}' + suffix[6:]
        histname = prefix + suffix
        return histname

    def class_limit(arr, classes_to_keep):
        hist = arr['hist']
        bg_intersection = arr['bg'][classes_to_keep].sum()
        classes_to_keep = [0] + classes_to_keep
        hist = hist[:, classes_to_keep]
        hist = hist[classes_to_keep, :]
        hist[0, 0] = bg_intersection
        return hist

    all_scores = []
    iters = ['%03d000'% x for x in range(1,24) if x % 4 == 0]
    for it in iters:
        bin_ius = []
        for fold in range(4):
            fold_dir = result_dir.format(fold)
            hist = np.load(f"{fold_dir}/{make_histname(it, histname)}.npz")

            classes_to_keep = list(range(1 + fold * 5, 1 + (fold + 1) * 5))
            hist = class_limit(hist, classes_to_keep)

            metrics = SegScorer(21)
            metrics.hist = hist

            fold_scores = metrics.score()
            bin_ius.append(np.nanmean(fold_scores['pos_iu']))
        all_scores.append(np.mean(bin_ius))

    print('Max score:', max(all_scores), 'iter:', iters[np.argmax(all_scores)])

if __name__ == '__main__':
    main()
