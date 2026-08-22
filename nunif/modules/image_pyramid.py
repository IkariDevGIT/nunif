import torch
import torch.nn.functional as F


def ensure_4d(x: torch.Tensor):
    is_3d = x.ndim == 3
    if is_3d:
        x = x.unsqueeze(0)
    return x, is_3d


def restore_dim(x: torch.Tensor, is_3d: bool):
    if is_3d:
        return x.squeeze(0)
    return x


def pyramid_decompose(x: torch.Tensor, num_levels: int) -> tuple[torch.Tensor]:
    assert num_levels > 1  # 2 == (1x, 2x)

    x, is_3d = ensure_4d(x)
    pyramid = []
    current = x

    for _ in range(num_levels - 1):
        # Use `antialias=True` instead of gaussian blur
        next_low = F.interpolate(current, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
        upsampled = F.interpolate(next_low, size=current.shape[-2:], mode="bilinear", align_corners=False)
        diff = current - upsampled
        pyramid.append(restore_dim(diff, is_3d))
        current = next_low

    pyramid.append(restore_dim(current, is_3d))

    return tuple(pyramid)


def pyramid_reconstruct(pyramid: list[torch.Tensor] | tuple[torch.Tensor]) -> torch.Tensor:
    current = pyramid[-1]
    _, is_3d = ensure_4d(current)

    for diff in reversed(pyramid[:-1]):
        current, _ = ensure_4d(current)
        diff, _ = ensure_4d(diff)
        upsampled = F.interpolate(current, size=diff.shape[-2:], mode="bilinear", align_corners=False)
        current = upsampled + diff
        current = restore_dim(current, is_3d)

    return current


def _test():
    x = torch.rand((4, 3, 512, 512)).double().cuda()

    p1 = pyramid_decompose(x, 2)
    p2 = pyramid_decompose(x, 3)
    p3 = pyramid_decompose(x, 4)

    assert len(p1) == 2
    assert len(p2) == 3
    assert len(p3) == 4

    assert p3[0].shape == (4, 3, 512, 512)
    assert p3[1].shape == (4, 3, 256, 256)
    assert p3[2].shape == (4, 3, 128, 128)
    assert p3[3].shape == (4, 3, 64, 64)

    r1 = pyramid_reconstruct(p1)
    r2 = pyramid_reconstruct(p2)
    r3 = pyramid_reconstruct(p3)

    assert torch.all(torch.isclose(x, r1))
    assert torch.all(torch.isclose(x, r2))
    assert torch.all(torch.isclose(x, r3))


def _test_laplacian_blend():
    import torchvision.transforms.functional as TF
    from torchvision.io import read_image

    from .gaussian_filter import gaussian_blur2d

    img1 = (read_image("tmp/apple.jpg") / 255).unsqueeze(0)
    img2 = (read_image("tmp/orange.jpg") / 255).unsqueeze(0)

    mask = torch.zeros_like(img1)
    mask[..., : mask.shape[-1] // 2] = 1.0

    octaves = 4
    k_size = 5

    if False:
        mask = gaussian_blur2d(mask, kernel_size=k_size)

    pry1 = pyramid_decompose(img1, octaves)
    pyr2 = pyramid_decompose(img2, octaves)

    pyr_mask = []
    current_mask = mask
    for i in range(octaves):
        pyr_mask.append(current_mask)
        blurred_mask = gaussian_blur2d(current_mask, kernel_size=k_size)
        current_mask = F.interpolate(
            blurred_mask, scale_factor=0.5, mode="bilinear", antialias=True, align_corners=False
        )

    pyr_mask.append(current_mask)

    pyr_blended = []
    for p1, p2, m in zip(pry1, pyr2, pyr_mask):
        blended_layer = p1 * m + p2 * (1.0 - m)
        pyr_blended.append(blended_layer)

    img_blended = pyramid_reconstruct(pyr_blended)
    TF.to_pil_image(img_blended[0].clamp(0, 1)).show()


if __name__ == "__main__":
    _test()
    # _test_laplacian_blend()
