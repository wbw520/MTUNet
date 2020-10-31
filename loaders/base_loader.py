import torch
from torch.utils.data import DataLoader, Dataset
import loaders.datasets as datasets


def get_dataloader(args, split, shuffle=True, out_name=False, sample=None, selection=None, mode=None):
    # sample: iter, way, shot, query
    transform = datasets.make_transform(args, split)
    sets = datasets.DatasetFolder(args.data_root, args.dataset, split, transform, out_name=out_name, cls_selction=selection, mode=mode)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.num_workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.num_workers, pin_memory=True)
    return loader
