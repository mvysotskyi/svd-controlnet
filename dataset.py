import os, random
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


def pil_image_to_numpy(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


def find_image_folders(root_dir):
    image_folders = []
    image_extensions = ['.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']
    
    for dirpath, _, filenames in os.walk(root_dir):
        if not filenames:
            continue
        print(dirpath)
    
        is_image_folder = all(
            any(filename.lower().endswith(ext) for ext in image_extensions)
            for filename in filenames
        )

        if is_image_folder and filenames:
            image_folders.append(dirpath)
    
    return image_folders


def get_crop_size(image, target_size):
    width, height = image.size
    crop_size = min(width, height)
    
    if crop_size < target_size:
        return None
    
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    
    return (left, top, right, bottom)

def random_crop_resize(image, crop_size, target_size):
    left, top, right, bottom = crop_size
    
    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize(target_size, Image.BILINEAR)
    
    return image_resized




class VideoDataset(Dataset):
    def __init__(
            self, dataset_root_dir, image_size=256
        ):
        self.paths = find_image_folders(dataset_root_dir)
        print(self.paths)
        self.images_size = image_size

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])

    def sort_frames(frame_name):
        return int(frame_name.split(".")[0])
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        video_frames_path = self.paths[idx]
        if "depth" in video_frames_path:
            idx = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(idx)
        
        image_files = sorted(os.listdir(video_frames_path), key=__class__.sort_frames)
        image_files = [os.path.join(video_frames_path, image_file) for image_file in image_files]
        random_shift = random.randint(0, len(image_files) - 14)
        image_files = image_files[random_shift:random_shift + 14]

        depth_path = video_frames_path + "_depth"
        depth_files = sorted(os.listdir(depth_path), key=__class__.sort_frames)[random_shift:random_shift + 14]

        if len(image_files) < 14:
            idx = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(idx)

        pil_images = [Image.open(image_path) for image_path in image_files]
        depth_images = [Image.open(os.path.join(depth_path, depth_file)) for depth_file in depth_files]
        if any(min(image.size) < self.images_size for image in pil_images):
            idx = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(idx)
        
        crop_size = get_crop_size(pil_images[0], self.images_size)
        if crop_size is None:
            idx = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(idx)

        pixel_values = np.array([np.array(random_crop_resize(image, crop_size, (self.images_size, self.images_size))) for image in pil_images])
        canny_images = np.array([np.stack([np.array(random_crop_resize(image, crop_size, (self.images_size, self.images_size)))]*3, axis=-1) for image in depth_images])

        return dict(pixel_values=numpy_to_pt(pixel_values), canny=numpy_to_pt(canny_images), motion_values=127.0)


if __name__ == "__main__":
    dataset = VideoDataset("../bear-1", image_size=256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data['pixel_values'].shape)
        print(data['canny'].shape)
        print(data['motion_values'].shape)
        break