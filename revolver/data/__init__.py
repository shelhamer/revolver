import os
import pickle

import torch
from torchvision.transforms import Compose

from .pascal import *
from .coco import COCO
from .seg import MaskInstSeg, MaskSemSeg
from .sparse import SparseSeg
from .interactive import InteractiveSeg
from .filter import TargetFilter, TargetMapper, SubSampler, MultiFilter
from .conditional import ConditionalInstSeg, ConditionalSemSeg
from .crop import CropSeg
from .class_balance import ClassBalance
from .transforms import *
from .util import InputsTargetAuxCollate


def prepare_semantic_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if classes_to_filter is not None:
        raise Exception("Cannot load semantic data with class groups.")
    if count is not None:
        raise Exception("Cannot load semantic data with sparsity.")
    cache_path = make_cache_path('semantic', dataset_name, split, classes_to_filter=None, count=-1, multi=multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    dataset = datasets[dataset_name]
    ds = dataset(split=split)
    if multi:
        ds = MultiFilter(ds)
    image_transform = Compose([
        ImToCaffe(mean=dataset.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(ds, input_transforms=[image_transform],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_fgbg_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if count is not None:
        raise Exception("Cannot load fg-bg data with sparsity.")
    cache_path = make_cache_path('fgbg', dataset_name, split, classes_to_filter, count=-1, multi=multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    dataset = datasets[dataset_name]
    ds = dataset(split=split)
    if multi:
        ds = MultiFilter(ds)
    filter_ds = filter_classes(ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_ds, classes_to_filter)
    # map all included classes (except background) to positive (== 1)
    classes_to_filter = classes_to_filter or range(1, len(ds.classes))
    # map must include bg so that no. of classes == 2
    fgbg_ds = TargetMapper(class_bal_ds,
        {k: int(k > 0) if k in classes_to_filter else 0 for k in range(len(filter_ds.classes))})
    image_transform = Compose([
        ImToCaffe(mean=dataset.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(fgbg_ds, input_transforms=[image_transform],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_interactive_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if multi:
        raise Exception("Cannot multi interactive instance.")
    cache_path = make_cache_path('interactive', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    inst_dataset = datasets[dataset_name + '-inst']
    inst_ds = inst_dataset(split=split)
    mask_ds = MaskInstSeg(sem_ds, inst_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    sparse_ds = sparsify(class_bal_ds, count=count)
    inter_ds = InteractiveSeg(class_bal_ds, sparse_ds)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(inter_ds,
        input_transforms=[image_transform, None],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_interactive_class_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    cache_path = make_cache_path('interactive-class', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    if multi:
        sem_ds = MultiFilter(sem_ds)
    mask_ds = MaskSemSeg(sem_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    sparse_ds = sparsify(class_bal_ds, count=count)
    inter_ds = InteractiveSeg(class_bal_ds, sparse_ds)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(inter_ds,
        input_transforms=[image_transform, None],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_early_interactive_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if multi:
        raise Exception("Cannot multi interactive instance.")
    cache_path = make_cache_path('early-interactive', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    inst_dataset = datasets[dataset_name + '-inst']
    inst_ds = inst_dataset(split=split)
    mask_ds = MaskInstSeg(sem_ds, inst_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    sparse_ds = sparsify(class_bal_ds, count)
    inter_ds = InteractiveSeg(class_bal_ds, sparse_ds)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    anno_transform = Compose([DilateMask(11), ScaleMask(256)])
    transform_ds = TransformData(inter_ds,
        input_transforms=[image_transform, anno_transform],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_early_interactive_class_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    cache_path = make_cache_path('early-interactive-class', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    if multi:
        sem_ds = MultiFilter(sem_ds)
    mask_ds = MaskSemSeg(sem_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    sparse_ds = sparsify(class_bal_ds, count)
    inter_ds = InteractiveSeg(class_bal_ds, sparse_ds)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    anno_transform = Compose([DilateMask(11), ScaleMask(256)])
    transform_ds = TransformData(inter_ds,
        input_transforms=[image_transform, anno_transform],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_early_conditional_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if multi:
        raise Exception("Cannot multi early conditional instance data.")
    cache_path = make_cache_path('early-conditional', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        ds = pickle.load(open(cache_path, 'rb'))
        ds.ds.shot = shot  # don't worry about it...
        return ds
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    inst_dataset = datasets[dataset_name + '-inst']
    inst_ds = inst_dataset(split=split)
    mask_ds = MaskInstSeg(sem_ds, inst_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    crop_ds = CropSeg(class_bal_ds)
    sparse_ds = sparsify(crop_ds, count=count)
    cond_ds = ConditionalInstSeg(class_bal_ds, sparse_ds, shot=shot)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    anno_transform = Compose([DilateMask(11), ScaleMask(256)])
    transform_ds = TransformData(cond_ds,
        input_transforms=[image_transform, (image_transform, anno_transform)],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_early_conditional_class_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    cache_path = make_cache_path('early-conditional-class', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        ds = pickle.load(open(cache_path, 'rb'))
        ds.ds.shot = shot  # don't worry about it...
        return ds
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    if multi:
        sem_ds = MultiFilter(sem_ds)
    mask_ds = MaskSemSeg(sem_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    if classes_to_filter is None:
        class_bal_ds = filter_mask_ds
        support_datasets = [TargetFilter(filter_mask_ds, [c]) for c in range(1, len(sem_ds.classes))]
    else:
        class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
        support_datasets = class_bal_ds.datasets  # rude, but simple
    sparse_datasets = [sparsify(ds, count=count) for ds in support_datasets]
    cond_ds = ConditionalSemSeg(class_bal_ds, sparse_datasets, shot=shot)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    anno_transform = Compose([DilateMask(11), ScaleMask(256)])
    transform_ds = TransformData(cond_ds,
        input_transforms=[image_transform, (image_transform, anno_transform)],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_late_conditional_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    if multi:
        raise Exception("Cannot multi late conditional instance data.")
    cache_path = make_cache_path('late-conditional', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        ds = pickle.load(open(cache_path, 'rb'))
        ds.ds.shot = shot  # don't worry about it...
        return ds
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    inst_dataset = datasets[dataset_name + '-inst']
    inst_ds = inst_dataset(split=split)
    mask_ds = MaskInstSeg(sem_ds, inst_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
    crop_ds = CropSeg(class_bal_ds)
    sparse_ds = sparsify(crop_ds, count=count)
    cond_ds = ConditionalInstSeg(class_bal_ds, sparse_ds, shot=shot)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(cond_ds,
        input_transforms=[image_transform, (image_transform, None)],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_late_conditional_class_data(dataset_name, split, classes_to_filter=None, count=None, shot=None, multi=False):
    cache_path = make_cache_path('late-conditional-class', dataset_name, split, classes_to_filter, count, multi)
    if os.path.isfile(cache_path):
        ds = pickle.load(open(cache_path, 'rb'))
        ds.ds.shot = shot  # don't worry about it...
        return ds
    sem_dataset = datasets[dataset_name]
    sem_ds = sem_dataset(split=split)
    if multi:
        sem_ds = MultiFilter(sem_ds)
    mask_ds = MaskSemSeg(sem_ds)
    filter_mask_ds = filter_classes(mask_ds, classes_to_filter)
    if classes_to_filter is None:
        class_bal_ds = filter_mask_ds
        support_datasets = [TargetFilter(filter_mask_ds, [c]) for c in range(1, len(sem_ds.classes))]
    else:
        class_bal_ds = balance_classes(filter_mask_ds, classes_to_filter)
        support_datasets = class_bal_ds.datasets  # rude, but simple
    sparse_datasets = [sparsify(ds, count=count) for ds in support_datasets]
    cond_ds = ConditionalSemSeg(class_bal_ds, sparse_datasets, shot=shot)
    image_transform = Compose([
        ImToCaffe(mean=sem_ds.mean),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    transform_ds = TransformData(cond_ds,
        input_transforms=[image_transform, (image_transform, None)],
        target_transform=target_transform)
    pickle.dump(transform_ds, open(cache_path, 'wb'))
    return transform_ds

def prepare_loader(dataset, evaluation=False):
    shuffle = True
    num_workers = 4
    if evaluation:
        shuffle = False
        num_workers = 0  # for determinism
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle,
        collate_fn=InputsTargetAuxCollate(),
        num_workers=num_workers, pin_memory=True)

def make_cache_path(datatype, dataset_name, split, classes_to_filter=None, count=None, multi=False):
    classes_to_filter = '-'.join(str(c) for c in classes_to_filter) if classes_to_filter else 'all'
    count = 'dense' if count == -1 else "{}sparse".format(count) if count else 'randsparse'
    multi = 'multi' if multi else ''
    return "{__CACHE_DIR__}/{datatype}_{dataset_name}_{split}_{classes_to_filter}_{count}_{multi}.pkl".format(
            __CACHE_DIR__=__CACHE_DIR__, datatype=datatype, dataset_name=dataset_name, split=split, classes_to_filter=classes_to_filter, count=count, multi=multi)

def filter_classes(ds, classes_to_keep=None):
    if classes_to_keep:
        # filter to keep all elements with any class to keep
        ds = TargetFilter(ds, classes_to_keep)
    return ds

def balance_classes(ds, classes_to_keep=None):
    # note: only balance class groups, and not all classes!
    if classes_to_keep is None:
        return ds
    class_datasets = [TargetFilter(ds, [c]) for c in classes_to_keep]
    num_sample = min([len(ds) for ds in class_datasets])
    balanced_datasets = [SubSampler(ds, num_sample) for ds in class_datasets]
    class_bal_ds = ClassBalance(balanced_datasets)
    return class_bal_ds

def sparsify(ds, count=None):
    if count is None:
        count = list(range(100))
    return SparseSeg(ds, count=count)

datasets = {
    'voc': VOCSemSeg,
    'sbdd': SBDDSemSeg,
    'voc-inst': VOCInstSeg,
    'sbdd-inst': SBDDInstSeg,
    'coco': COCO,
}

datatypes = {
    'semantic': prepare_semantic_data,
    'fgbg': prepare_fgbg_data,
    'interactive': prepare_interactive_data,
    'interactive-class': prepare_interactive_class_data,
    'early-interactive': prepare_early_interactive_data,
    'early-interactive-class': prepare_early_interactive_class_data,
    'early-conditional': prepare_early_conditional_data,
    'early-conditional-class': prepare_early_conditional_class_data,
    'late-conditional': prepare_late_conditional_data,
    'late-conditional-class': prepare_late_conditional_class_data,
}

# make cache dir for datasets with intensive init
__CACHE_DIR__ = './data/cache'
try:
    os.makedirs(__CACHE_DIR__, exist_ok=True)
except:
    raise Exception("Could not create cache dir {}".format(__CACHE_DIR__))
