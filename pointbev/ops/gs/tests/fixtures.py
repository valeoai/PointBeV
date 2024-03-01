import pytest
import torch

torch.set_printoptions(precision=12)
N, S, C, D, H, W = 1, 6, 128, 1, 28, 60
Do, Ho, Wo = 1, 1, 200 * 200 * 8


@pytest.fixture(params=[0.0, 0.16, 0.25, 0.5, 1.0])
def set_pct_mask(request):
    return request.param


@pytest.fixture
def input_data(set_pct_mask):
    device = "cuda"

    rand = torch.randn(N, S, C, D, H, W, device=device)
    grid = torch.rand(N, S, Do, Ho, Wo, 3, device=device, requires_grad=False) * 2 - 1
    mask = (
        torch.rand(N, S, Do, Ho, Wo, device=device, requires_grad=False)
        .le(set_pct_mask)
        .float()
    )

    return (
        rand,
        grid,
        mask,
        {"pct": f"mask:{set_pct_mask}"},
    )
