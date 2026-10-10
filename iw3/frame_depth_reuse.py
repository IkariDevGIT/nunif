from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn.functional as F


class DepthPredictor(Protocol):
    model: Any

    def infer(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...


def frames_are_similar(frame: torch.Tensor, reference: torch.Tensor, scale: torch.Tensor | None = None) -> bool:
    if frame.shape != reference.shape or frame.dtype != reference.dtype:
        return False
    reference = reference.to(frame.device)
    difference = frame.float() - reference.float()
    if scale is not None:
        difference = difference * scale
    # Full-resolution guards prevent a small moving object from disappearing in a global average.
    absolute = difference.abs()
    difference = difference.unsqueeze(0) if difference.ndim == 3 else difference
    local_bias = F.avg_pool2d(difference, 3, stride=1, padding=1, count_include_pad=False)
    mean_error = absolute.mean() if absolute.ndim == 3 else absolute.flatten(1).mean(1).amax()
    acceptable = (
        (absolute.amax() <= 6 / 255 + 1e-7) & (mean_error <= 1.25 / 255) & (local_bias.abs().amax() <= 2 / 255 + 1e-7)
    )
    return bool(acceptable.item())


class FrameDepthReuse:
    def __init__(
        self, depth_model: DepthPredictor, comparison: Callable[[torch.Tensor, torch.Tensor], bool] = frames_are_similar
    ) -> None:
        self.depth_model = depth_model
        self.comparison = comparison
        self.frame: torch.Tensor | None = None
        self.depth: torch.Tensor | None = None
        self.key: tuple[Any, ...] | None = None
        self.ready: torch.cuda.Event | None = None

    def clear(self) -> None:
        self.frame = self.depth = self.key = self.ready = None

    def _wait(self) -> None:
        if self.ready is not None:
            assert self.frame is not None and self.depth is not None
            # Video workers may use different CUDA streams, even on the same device.
            for tensor in (self.frame, self.depth):
                if tensor.is_cuda:
                    stream = torch.cuda.current_stream(tensor.device)
                    stream.wait_event(self.ready)
                    tensor.record_stream(stream)

    @torch.inference_mode()
    def infer(
        self,
        x: torch.Tensor,
        *,
        comparison_frames: torch.Tensor | None = None,
        reset_after: list[bool] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batched = x.ndim == 4
        images = x if batched else x.unsqueeze(0)
        comparison = comparison_frames if comparison_frames is not None else x
        comparison = comparison if comparison.ndim == 4 else comparison.unsqueeze(0)
        if images.ndim != 4 or comparison.ndim != 4 or images.shape[0] != comparison.shape[0] or images.shape[0] == 0:
            raise ValueError("Frame reuse expects matching nonempty CHW/BCHW inputs")
        resets = reset_after if reset_after is not None else [False] * images.shape[0]
        if len(resets) != images.shape[0]:
            raise ValueError("Frame reuse scene boundaries must match the batch size")
        advance = getattr(self.depth_model, "advance_reused_frame", None)
        if batched and advance is not None:
            # Temporal models must observe fresh and reused frames in their original order.
            return torch.stack(
                [
                    self.infer(image, comparison_frames=frame, reset_after=[reset], **kwargs)
                    for image, frame, reset in zip(images, comparison, resets)
                ]
            )
        key = (
            id(self.depth_model.model),
            tuple(images.shape[1:]),
            images.dtype,
            tuple((name, tuple(value) if isinstance(value, list) else value) for name, value in sorted(kwargs.items())),
        )
        if key != self.key:
            self.clear()
        self._wait()
        reference = self.frame
        indices: list[int] = []
        mapping: list[int] = []
        slot = -1
        for index, frame in enumerate(comparison):
            if reference is None or not self.comparison(frame, reference):
                indices.append(index)
                slot = len(indices) - 1
                reference = frame
            mapping.append(slot)
            if resets[index]:
                reference = None

        predicted = None
        if indices:
            # Retain normal batching for changed frames; only remove redundant predictions.
            selected = images if len(indices) == len(images) else images[indices]
            predicted = self.depth_model.infer(selected if batched else selected[0], **kwargs)
            if not batched:
                predicted = predicted.unsqueeze(0)
        elif advance is not None:
            advance()
        previous = self.depth.to(images.device) if self.depth is not None else None
        example = predicted if predicted is not None else previous.unsqueeze(0) if previous is not None else None
        assert example is not None
        if predicted is not None and len(indices) == len(images):
            output = predicted
        else:
            output = torch.empty((len(images), *example.shape[1:]), dtype=example.dtype, device=example.device)
            for index, source in enumerate(mapping):
                value = previous if source == -1 else predicted[source] if predicted is not None else None
                assert value is not None
                output[index].copy_(value)

        if reference is None:
            self.clear()
            if advance is not None:
                getattr(self.depth_model, "reset_state")()
        elif indices:
            assert predicted is not None
            self.frame = reference.detach().clone()
            self.depth = predicted[slot].detach().clone()
            self.key = key
            if self.frame.is_cuda:
                self.ready = torch.cuda.Event()
                self.ready.record(torch.cuda.current_stream(self.frame.device))
        return output if batched else output[0]
