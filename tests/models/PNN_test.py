import pytest

from deepctr_torch.models import PNN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_device


@pytest.mark.parametrize(
    'use_inner, use_outter, kernel_type, sparse_feature_num',
    [(True, True, 'mat', 2), (True, False, 'mat', 2), (False, True, 'vec', 3), (False, True, 'num', 3),
     (False, False, 'mat', 1)
     ]
)
def test_PNN(use_inner, use_outter, kernel_type, sparse_feature_num):
    model_name = "PNN"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = PNN(feature_columns, dnn_hidden_units=[32, 32], dnn_dropout=0.5, use_inner=use_inner,
                use_outter=use_outter, kernel_type=kernel_type, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
