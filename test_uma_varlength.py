import torch
from uma import UMA


def test_uma_varlength():
    torch.manual_seed(0)
    batch = 3
    max_len = 12
    input_size = 16

    # Create inputs and varying lengths
    xs = torch.randn(batch, max_len, input_size)
    ilens = torch.tensor([12, 7, 3], dtype=torch.long)

    # Zero out padded frames to mimic real inputs
    for i in range(batch):
        if ilens[i] < max_len:
            xs[i, ilens[i]:] = 0.0

    uma = UMA(input_size=input_size, output_size=input_size)
    uma.eval()

    with torch.no_grad():
        out, olens = uma(xs, ilens)

    # Basic sanity checks
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert olens.numel() == batch, "olens should have one entry per batch"
    assert (olens >= 0).all(), "olens should be non-negative"

    print("test_uma_varlength passed: out.shape=", out.shape, "olens=", olens)


if __name__ == '__main__':
    test_uma_varlength()
