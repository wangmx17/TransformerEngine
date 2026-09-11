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
@pytest.mark.parametrize("full_step_count", [1, 2, 4])
def test_thd_out_correction_4_single(dtype, full_step_count):
    torch.manual_seed(1234)
    total_tokens = 16
    num_heads = 4
    dim_per_head = 64
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="musa")
    out_per_step = []
    lse_per_step = []

    for step in range(4):
        step_tokens = total_tokens if step < full_step_count else total_tokens // 2
        step_out = torch.randn(
            step_tokens, num_heads, dim_per_head, dtype=dtype, device="musa"
        )
        # Exercise the zero-preserving path used to avoid 0 * inf -> nan.
        step_out[::5] = 0
        out_per_step.append(step_out)
        lse_per_step.append(
            torch.randn(1, num_heads, step_tokens, dtype=torch.float32, device="musa")
        )

    expected_lse = lse_per_step[0].clone()
    for step in range(1, 4):
        if step < full_step_count:
            max_scale = torch.maximum(expected_lse, lse_per_step[step])
            min_scale = torch.minimum(expected_lse, lse_per_step[step])
            expected_lse.copy_(max_scale + torch.log(1 + torch.exp(min_scale - max_scale)))
        else:
            tex.thd_second_half_lse_correction(
                expected_lse, lse_per_step[step], cu_seqlens, False
            )

    expected_out = torch.zeros(
        total_tokens, num_heads, dim_per_head, dtype=dtype, device="musa"
    )
    for step in range(4):
        tex.thd_out_correction(
            expected_out,
            out_per_step[step],
            expected_lse,
            lse_per_step[step],
            cu_seqlens,
            step >= full_step_count,
            False,
        )

    actual_out = torch.empty_like(expected_out)
    actual_lse = lse_per_step[0].clone()
    tex.thd_out_correction_4_single(
        actual_out,
        out_per_step,
        actual_lse,
        lse_per_step,
        cu_seqlens,
        full_step_count,
    )

    tolerance = 1e-6 if dtype == torch.float32 else (1e-3 if dtype == torch.float16 else 1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_out, expected_out, rtol=tolerance, atol=tolerance)
