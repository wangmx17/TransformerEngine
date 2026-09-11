import pytest
import torch
import torch_musa

import transformer_engine  # noqa: F401  # Load TE libraries before importing the extension.
import transformer_engine_torch as tex


def _reference(lse, lse_per_step, cu_seqlens, lse_packed):
    expected = lse.clone()
    half_cu_seqlens = cu_seqlens.cpu() // 2
    batch = half_cu_seqlens.numel() - 1

    for seq_id in range(batch):
        half_start = int(half_cu_seqlens[seq_id])
        half_end = int(half_cu_seqlens[seq_id + 1])
        half_seq_len = half_end - half_start
        if lse_packed:
            expected[:, half_start + half_end : 2 * half_end] = torch.logaddexp(
                expected[:, half_start + half_end : 2 * half_end],
                lse_per_step[:, half_start:half_end].to(expected.dtype),
            )
        else:
            expected[seq_id, :, half_seq_len : 2 * half_seq_len] = torch.logaddexp(
                expected[seq_id, :, half_seq_len : 2 * half_seq_len],
                lse_per_step[seq_id, :, :half_seq_len].to(expected.dtype),
            )
    return expected


@pytest.mark.parametrize("lse_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lse_packed", [False, True])
def test_thd_second_half_lse_correction(lse_dtype, lse_packed):
    torch.manual_seed(1234)
    batch = 2
    num_heads = 4
    lse_seqlen = 12
    second_half_lse_seqlen = lse_seqlen // 2
    cu_seqlens = torch.tensor([0, 8, 12], dtype=torch.int32, device="musa")

    if lse_packed:
        lse = torch.randn(num_heads, lse_seqlen, dtype=lse_dtype, device="musa")
        lse_per_step = torch.randn(
            num_heads, second_half_lse_seqlen, dtype=torch.float32, device="musa"
        )
    else:
        lse = torch.randn(batch, num_heads, lse_seqlen, dtype=lse_dtype, device="musa")
        lse_per_step = torch.randn(
            batch,
            num_heads,
            second_half_lse_seqlen,
            dtype=torch.float32,
            device="musa",
        )

    expected = _reference(lse, lse_per_step, cu_seqlens, lse_packed)
    tex.thd_second_half_lse_correction(lse, lse_per_step, cu_seqlens, lse_packed)

    torch.testing.assert_close(lse, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("half_idx", [0, 1])
def test_thd_read_half_tensor_3(dtype, half_idx):
    torch.manual_seed(1234)
    cu_seqlens = torch.tensor([0, 8, 12], dtype=torch.int32, device="musa")
    tensors = [torch.randn(12, 4, 64, dtype=dtype, device="musa") for _ in range(3)]
    expected = [tex.thd_read_half_tensor(x, cu_seqlens, half_idx) for x in tensors]

    actual = tex.thd_read_half_tensor_3(*tensors, cu_seqlens, half_idx)

    assert len(actual) == 3
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
