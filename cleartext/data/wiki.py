import io
import os
import tarfile
from typing import Tuple

from torchtext.data import Example, Field
from torchtext.datasets import TranslationDataset

from .. import PROJ_ROOT


class WikiSL(TranslationDataset):
    """Torchtext loader for the WikiSmall and WikiLarge datasets."""
    urls = ['https://raw.githubusercontent.com/louismartin/dress-data/master/data-simplification.tar.bz2']

    # shadowed TranslationDataset.splits because its download classmethod doesn't extract bzip
    @classmethod
    def splits(cls, fields: Tuple[Field, Field], **kwargs)\
            -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
        """Create dataset object for the WikiSmall or WikiLarge dataset.

        :param fields: Tuple[Field, Field]
            Source and target fields, respectively.
        :param max_examples: int
            The maximum number of training examples to load.
        :param kwargs:
            Passed to TranslationDataset.splits.
        :return: Tuple[TranslationDataset, TranslationDataset, TranslationDataset]
            Training, validation, and testing datasets, respectively.
        """
        exts = ('.src', '.dst')
        root = PROJ_ROOT / 'data/raw/'

        train = cls.prefix + '.train'
        valid = cls.prefix + '.valid'
        test = cls.prefix + '.test'

        # download if necessary
        check = root / 'data-simplification'
        cls.download(root, check=check)

        # extract if necessary
        if not check.is_dir():
            url = cls.urls[0]
            path = root
            filename = os.path.basename(url)
            zpath = path / filename
            with tarfile.open(zpath, 'r:bz2') as tar:
                dirs = [member for member in tar.getmembers()]
                tar.extractall(path=root, members=dirs)

        path = check / cls.dir_name
        return super().splits(exts, fields, path=path, root=root, train=train, validation=valid, test=test, **kwargs)

    # shadowed super().__init__() to allow restricting the number of examples loaded (via kwargs)
    def __init__(self, path: str, exts: Tuple[str, str], fields: Tuple[Field, Field], **kwargs) -> None:
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(path + x for x in exts)

        max_examples = kwargs.get('max_examples')
        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for i, line in enumerate(zip(src_file, trg_file)):
                src_line, trg_line = line
                if max_examples and i == max_examples:
                    break
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields)


class WikiSmall(WikiSL):
    """The WikiSmall dataset."""

    name = ''
    dir_name = 'wikismall'
    dirname = ''
    prefix = 'PWKP_108016.tag.80.aner.ori'


class WikiLarge(WikiSL):
    """The WikiLarge dataset."""

    name = ''
    dir_name = 'wikilarge'
    dirname = ''
    prefix = 'wiki.full.aner.ori'
