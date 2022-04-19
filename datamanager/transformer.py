from torchvision import transforms
from typing import Any, List, Sequence, Callable
from PIL import Image, ImageFilter, ImageOps
import torch
import random
import attr
from pprint import pprint


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """
        return ImageOps.solarize(img)


## Multi-view Wrapper
class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"

class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


# public interface for build the data transformation
class Transform_builder(object):

    def __init__(self, dataset: str, kwargs={}) -> Any:
        """Prepares transforms for a specific dataset. Optionally uses multi crop.
        Args:
            dataset (str): name of the dataset.
        Returns:
            Any: a transformation for a specific dataset.
        """
        self.trfs_lst = []
        kwargs = [kwargs] if isinstance(kwargs, dict) else kwargs
        for args in kwargs:
            # limited with kw-args only
            if dataset in ['cifar10', 'cifar100']:
                self.trfs_lst.append( CifarTransform(cifar=dataset, **args) ) 
            elif dataset == "stl10":
                self.trfs_lst.append( STLTransform(**args) )  
            elif dataset == "slim":
                self.trfs_lst.append( SlimTransform(**args) )
            elif dataset in ['imagenet']: 
                self.trfs_lst.append(  ImagenetTransform(**args) )  
            elif dataset == "custom":
                self.trfs_lst.append(  StandardTransform(**args) )  
            else:
                raise ValueError(f"{dataset} is not currently supported.")
           
    def debug_transformation(self):
        print("Transforms:")
        pprint(self.trfs_lst)

    def prepare_n_crop_transform(
        self, num_crops_per_aug: List[int]
    ) -> NCropAugmentation:
        """Turns a single crop transformation to an N crops transformation.
        Args:
            transforms (List[Callable]): list of transformations.
            num_crops_per_aug (List[int]): number of crops per pipeline.
        Returns:
            NCropAugmentation: an N crop transformation.
        """

        assert len(num_crops_per_aug) <= 2, "The multi-crop is not supported currently"

        T = []
        for transform, num_crops in zip(self.trfs_lst, num_crops_per_aug):
            T.append( NCropAugmentation(transform, num_crops) )
        return FullTransformPipeline(T)


## Build Transform Pipeline
class BaseTransform:
    def __init__(self, trfs_lst:list=[]):
        self.transform = transforms.Compose(trfs_lst)

    """Adds callable base class to implement different transformation pipelines."""
    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)


@attr.s(auto_attribs=True, kw_only=True)
class StandardTransform(BaseTransform):
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    color_jitter_prob: float = 0.8
    gray_scale_prob: float = 0.2
    horizontal_flip_prob: float = 0.5
    gaussian_prob: float = 0.5
    solarization_prob: float = 0.0
    min_scale: float = 0.08
    max_scale: float = 1.0
    crop_size: int = 224
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.228, 0.224, 0.225)
    """Class that applies Custom transformations.
    If you want to do exoteric augmentations, you can just re-write this class.

    Args:
        brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            Defaults to 0.4.
        contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            Defaults to 0.4.
        saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            Defaults to 0.2.
        hue (float): sampled uniformly in [-hue, hue].
            Defaults to 0.1.
        color_jitter_prob (float, optional): probability of applying color jitter.
            Defaults to 0.8.
        gray_scale_prob (float, optional): probability of converting to gray scale.
            Defaults to 0.2.
        horizontal_flip_prob (float, optional): probability of flipping horizontally.
            Defaults to 0.5.
        gaussian_prob (float, optional): probability of applying gaussian blur.
            Defaults to 0.0.
        solarization_prob (float, optional): probability of applying solarization.
            Defaults to 0.0.
        min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        crop_size (int, optional): size of the crop. Defaults to 224.
        mean (Sequence[float], optional): mean values for normalization.
            Defaults to (0.485, 0.456, 0.406).
        std (Sequence[float], optional): std values for normalization.
            Defaults to (0.228, 0.224, 0.225).
    """
    def __attrs_post_init__(self):
        trfs_lst = [
            transforms.RandomResizedCrop(
                self.crop_size,
                scale=(self.min_scale, self.max_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)],
                p=self.color_jitter_prob,
            ),
            transforms.RandomGrayscale(p=self.gray_scale_prob),
            transforms.RandomApply([GaussianBlur()], p=self.gaussian_prob),
            transforms.RandomApply([Solarization()], p=self.solarization_prob),
            transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ] # BaseTransform will wrap it into the transform.Composer
        super().__init__(trfs_lst)


@attr.s(auto_attribs=True)
class CifarTransform(StandardTransform):
    """Class that applies Cifar10/Cifar100 transformations.
    Args:
        cifar (str): type of cifar, either cifar10 or cifar100.
        crop_size (int, optional): size of the crop. Defaults to 32.
    """
    cifar: str
    crop_size: int = 32
    
    def __post_init__(self):
        if self.cifar == "cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
        else:
            self.mean = (0.5071, 0.4865, 0.4409)
            self.std = (0.2673, 0.2564, 0.2762)
        # initial the BaseTransformer
        super().__attrs_post_init__()


@attr.s(auto_attribs=True)
class STLTransform(StandardTransform):
    """Class that applies STL10 transformations.
    Args:
        crop_size (int, optional): size of the crop. Defaults to 96.
    """
    crop_size: int = 96

    def __post_init__(self):
        self.mean = (0.4914, 0.4823, 0.4466)
        self.std = (0.247, 0.243, 0.261)
        super().__attrs_post_init__()


@attr.s(auto_attribs=True)
class SlimTransform(StandardTransform):
    """Class that applies Slimagenet transformations.
    Args:
        crop_size (int, optional): size of the crop. Defaults to 96.
    """
    crop_size: int = 64

    def __post_init__(self):
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.25, 0.25, 0.25)
        super().__attrs_post_init__()


@attr.s(auto_attribs=True)
class ImagenetTransform(StandardTransform):
    """Class that applies Imagenet transformations.
    Args:
        crop_size (int, optional): size of the crop. Defaults to 224.
    """
    crop_size: int = 224
    
    def __post_init__(self):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.228, 0.224, 0.225)
        super().__attrs_post_init__()


if __name__ == "__main__":
    c = CifarTransform(cifar='cifar100', hue=0.4)
    print(c)