from torch.nn import Module

from cleartext.utils.utils import count_parameters


def test_count_parameters():
    model = Module()
    trainable, total = count_parameters(model)
    assert trainable == 0 and total == 0
