import torch


def rotate_reference(x, theta):
    # Naive and readable implementation
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2

    sin = theta.sin()
    cos = theta.cos()

    point_x = x[:half]
    point_y = x[half:]
    rotated_point_x = point_x * cos - point_y * sin
    rotated_point_y = point_x * sin + point_y * cos
    output = torch.cat([rotated_point_x, rotated_point_y], dim=-1)

    return output


def rotate_with_rotate_half(x, theta):
    # Commonly used implementation
    assert x.shape[0] % 2 == 0

    theta_repeated = torch.cat([theta, theta], dim=-1)
    cos = theta_repeated.cos()
    sin = theta_repeated.sin()

    def rotate_half(x):
        half = x.shape[0] // 2
        x1 = x[:half]
        x2 = x[half:]
        return torch.cat((-x2, x1), dim=-1)

    output = x * cos + rotate_half(x) * sin

    return output


def rotate_with_rotate_half_inplace(x, theta):
    # If you look properly, you'll see that it's the same as `rotate_reference`.
    # And this implementation is faster then `rotate_with_rotate_half` in eager mode.
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2

    cos = theta.cos()
    sin = theta.sin()

    output = torch.empty_like(x)
    x1 = x[:half]
    x2 = x[half:]
    output[:half] = x1 * cos - x2 * sin
    output[half:] = x2 * cos + x1 * sin

    return output


def rotate_with_rotate_half_inplace2(x, theta):
    # This is faster than `rotate_with_rotate_half_inplace` in eager mode.
    # However, it fails to compile with Torch 2.12. It appears to have been fixed in 2.13.
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2

    cos = theta.cos()
    cos = torch.cat([cos, cos], dim=-1)
    sin = theta.sin()

    output = x * cos
    output[:half].addcmul_(x[half:], sin, value=-1)
    output[half:].addcmul_(x[:half], sin, value=1)

    return output.to(x.dtype)


def transpose_wrapper(f):
    # A wrapper for converting between halving and odd-even splitting

    def _wrapper(*args, **kwargs):
        x, theta = args
        x_transposed = x.reshape((2, -1)).permute(1, 0).reshape(-1).contiguous()
        output_transposed = f(x_transposed, theta)
        output = output_transposed.reshape(-1, 2).permute(1, 0).reshape((-1,)).contiguous()
        return output

    _wrapper.__name__ = f.__name__

    return _wrapper


@transpose_wrapper
def rotate_with_complex(x, theta):
    # Complex number implementation
    # If optimized, this is probably the fastest option in eager mode, but it doesn't support `torch.comple`.
    theta_i = torch.polar(torch.ones_like(theta), theta)
    x_i = torch.view_as_complex(x.reshape(-1, 2))
    output = torch.view_as_real(x_i * theta_i).reshape(-1)

    return output


def main():
    N = 16
    assert N % 2 == 0
    x = torch.linspace(0, 1, N)
    theta = torch.linspace(0, torch.pi * 2, N // 2)

    reference = rotate_reference(x, theta)
    print("rotate_reference\n", reference)
    for f in [
        rotate_with_rotate_half,
        rotate_with_rotate_half_inplace,
        rotate_with_rotate_half_inplace2,
        rotate_with_complex,
    ]:
        z = f(x, theta)
        print(f"{f.__name__}\n", z)
        check = torch.all(torch.isclose(reference, z))
        print(f"rotate_reference == {f.__name__}", check)


if __name__ == "__main__":
    main()
