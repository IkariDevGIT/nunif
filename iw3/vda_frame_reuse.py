from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from .frame_depth_reuse import FrameDepthReuse, frames_are_similar


class DelayedDepthReuse:
    def __init__(self) -> None:
        self.reference: torch.Tensor | None = None
        self.depth: torch.Tensor | None = None
        self.pending: deque[bool] = deque()

    def clear(self) -> None:
        self.reference = self.depth = None
        self.pending.clear()

    def append(self, frame: torch.Tensor) -> bool:
        reuse = self.reference is not None and frames_are_similar(frame, self.reference)
        self.pending.append(reuse)
        if not reuse:
            self.reference = frame.detach().clone()
        return reuse

    def apply(self, depths: torch.Tensor) -> torch.Tensor:
        output = []
        for depth in depths:
            reuse = self.pending.popleft()
            if not reuse:
                self.depth = depth.detach().clone()
            assert self.depth is not None
            output.append(self.depth)
        return torch.stack(output)


class _FeaturePredictor:
    def __init__(self) -> None:
        self.model: Any = None
        self.get_features: Any = None

    def infer(self, x: torch.Tensor, *, amp: bool = False, **kwargs: Any) -> torch.Tensor:
        features = self.get_features(x, return_class_token=True, **kwargs)
        return torch.stack([torch.cat((patches, token.unsqueeze(1)), dim=1) for patches, token in features], dim=1)


class VDAEncoderReuse:
    def __init__(self) -> None:
        self.predictor = _FeaturePredictor()
        # Decoded-frame guards handle approximate reuse; encoder inputs use exact matching
        # so resizing cannot hide a real small change in a mixed temporal window.
        self.cache = FrameDepthReuse(self.predictor, comparison=torch.equal)

    def clear(self) -> None:
        self.cache.clear()

    @contextmanager
    def context(self, encoder: Any) -> Iterator[None]:
        self.predictor.model = encoder
        self.predictor.get_features = encoder.get_intermediate_layers
        # Compiled modules forward custom methods to their original module.
        target = getattr(encoder, "_orig_mod", encoder)
        previous = target.__dict__.get("get_intermediate_layers")

        def get_intermediate_layers(
            x: torch.Tensor,
            n: int | list[int] = 1,
            reshape: bool = False,
            return_class_token: bool = False,
            norm: bool = True,
        ) -> tuple[Any, ...]:
            if reshape:
                return self.predictor.get_features(
                    x, n=n, reshape=reshape, return_class_token=return_class_token, norm=norm
                )
            features = self.cache.infer(
                x,
                n=n,
                norm=norm,
                amp=torch.is_autocast_enabled(x.device.type),
            )
            if return_class_token:
                return tuple((layer[:, :-1], layer[:, -1]) for layer in features.unbind(1))
            return tuple(layer[:, :-1] for layer in features.unbind(1))

        target.get_intermediate_layers = get_intermediate_layers
        try:
            yield
        finally:
            if previous is None:
                del target.get_intermediate_layers
            else:
                target.get_intermediate_layers = previous


class VDAInferenceReuse(VDAEncoderReuse):
    def __init__(self) -> None:
        super().__init__()
        self.window: torch.Tensor | None = None
        self.prediction: torch.Tensor | None = None
        self.amp: bool | None = None

    def clear_window(self) -> None:
        self.window = self.prediction = self.amp = None

    def clear(self) -> None:
        super().clear()
        self.clear_window()

    @contextmanager
    def context(self, model: Any) -> Iterator[None]:
        original = model.forward
        previous = model.__dict__.get("forward")

        def forward(x: torch.Tensor) -> torch.Tensor:
            amp = torch.is_autocast_enabled(x.device.type)
            scale = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            if self.window is not None and self.prediction is not None and amp == self.amp:
                if frames_are_similar(x.flatten(0, 1), self.window.flatten(0, 1), scale=scale):
                    return self.prediction.clone()
            out = original(x)
            self.window = x.detach().clone()
            self.prediction = out.detach().clone()
            self.amp = amp
            return out

        model.forward = forward
        try:
            with super().context(model.pretrained):
                yield
        finally:
            if previous is None:
                del model.forward
            else:
                model.forward = previous
