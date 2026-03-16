import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader

info = lambda x: print(type(x), x.shape, x.dtype, x.min(), x.max())

class ImageSegDataset(Dataset):

    def __init__(self, image_paths, mask_paths, resize=(512, 512), is_train=False):
        """
        img_paths : list of RGB image paths
        mask_paths: list of mask paths (0/255 or 0/1)
        resize    : (H, W) 统一尺寸
        is_train  : 是否是训练集, 默认False
        """
        self.img_paths = image_paths
        self.msk_paths = mask_paths

        if is_train:
            # 训练集
            self.transform = A.Compose([
                    A.Resize(height=int(resize[0] * 1.2), width=int(resize[1] * 1.2), p=1.0),
                    A.Affine(
                        rotate=(-10, 10),
                        translate_percent=(-0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=(-5, 5),
                        p=0.5
                    ),
                    A.ElasticTransform(alpha=100, sigma=5, p=0.3),
                    A.RandomCrop(height=resize[0], width=resize[1], pad_if_needed=True),
                    A.pytorch.ToTensorV2()
                ],
                additional_targets={'mask': 'image'}
            )
        else:
            # 验证/测试集
            self.transform = A.Compose([
                    A.Resize(height=resize[0], width=resize[1], p=1.0),
                    A.pytorch.ToTensorV2()
                ],
                additional_targets={'mask': 'image'}
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msk_paths[idx], cv2.IMREAD_GRAYSCALE)
        msk = (msk > 0).astype(np.uint8)

        trans = self.transform(image=img, mask=msk)
        img, msk = trans['image'], trans['mask']

        # img: [3, H, W] torch.uint8 0-255
        # msk: [1, H, W] torch.uint8 0/1
        return img, msk


def read_index_csv(csv_path):
    """ 读取数据集图片标签文件路径 """
    image_paths, mask_paths = pd.read_csv(csv_path, header=None).values.T
    return image_paths, mask_paths


if __name__ == '__main__':
    
    img_paths, msk_paths = read_index_csv('data/idx_FIVES.csv')

    data_loader = DataLoader(
        ImageSegDataset(img_paths, msk_paths, is_train=False),
        batch_size=1, shuffle=True
    )

    for images, masks in data_loader:
        # batch_images = images.float() / 255.0
        # batch_masks = masks.float()
        # print(batch_images.shape, batch_masks.shape)
        batch_images = images
        batch_masks = masks * 255
        break
    
    from matplotlib import pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(batch_images[0].permute(1, 2, 0))
    
    plt.subplot(1, 2, 2)
    plt.imshow(batch_masks[0].permute(1, 2, 0), cmap='gray')

    plt.show()