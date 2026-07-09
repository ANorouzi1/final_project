import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


class Identity:
    def __call__(self, sample):
        return sample

    def __repr__(self):
        return "Identity()"


class FTWTorchvisionAugmentation:
    """Torchvision-backed augmentations for image/mask/SDF FTW samples."""

    def __init__(self):
        self.geometry = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([
                v2.RandomRotation(
                    degrees=(-45, 45),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            ], p=0.75),
            v2.RandomApply([
                v2.RandomAffine(
                    degrees=0,
                    translate=(0.06, 0.06),
                    scale=(0.90, 1.10),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            ], p=0.45),
        ])
        self.rgb_jitter = v2.ColorJitter(brightness=0.12, contrast=0.12)
        self.photometric = v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
            v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.025, clip=True)], p=0.35),
            v2.RandomErasing(p=0.25, scale=(0.005, 0.03), ratio=(0.5, 2.0), value=0.0),
        ])

    def __call__(self, sample):
        image = tv_tensors.Image(torch.as_tensor(sample["image"]).float())
        mask = tv_tensors.Mask(torch.as_tensor(sample["mask"]))
        distance = tv_tensors.Image(torch.as_tensor(sample["distance"]).float())
        sdf = tv_tensors.Image(torch.as_tensor(sample["sdf"]).float())
        instance = tv_tensors.Mask(torch.as_tensor(sample["instance"]))

        image, mask, distance, sdf, instance = self.geometry(
            image,
            mask,
            distance,
            sdf,
            instance,
        )

        image = self._jitter_rgb_windows(image)
        image = tv_tensors.Image(self.photometric(image).clamp(0.0, 1.0))

        out = dict(sample)
        out["image"] = _to_numpy(image).astype("float32", copy=False)
        out["mask"] = _to_numpy(mask).astype(sample["mask"].dtype, copy=False)
        out["distance"] = _to_numpy(distance).astype("float32", copy=False)
        out["sdf"] = _to_numpy(sdf).astype("float32", copy=False)
        out["instance"] = _to_numpy(instance).astype(sample["instance"].dtype, copy=False)
        return out

    def _jitter_rgb_windows(self, image):
        tensor = image.as_subclass(torch.Tensor).clone()
        for start in (0, 4):
            stop = start + 3
            if tensor.shape[0] >= stop:
                rgb = tv_tensors.Image(tensor[start:stop])
                tensor[start:stop] = self.rgb_jitter(rgb).as_subclass(torch.Tensor)
        return tv_tensors.Image(tensor.clamp(0.0, 1.0))

    def __repr__(self):
        return (
            "FTWTorchvisionAugmentation("
            "RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, "
            "RandomAffine, ColorJitter, GaussianBlur, GaussianNoise, RandomErasing)"
        )


def _to_numpy(value):
    if hasattr(value, "as_subclass"):
        value = value.as_subclass(torch.Tensor)
    return value.detach().cpu().numpy()


presets = dict(
    FTW=dict(
        train=Identity(),
        eval=Identity(),
    ),
    FTW_WithAugmentation=dict(
        train=FTWTorchvisionAugmentation(),
        eval=Identity(),
    ),
)
