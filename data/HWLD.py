import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class HandwritingLineDataset(Dataset):
    def __init__(self, csv_path, img_height=32, transform=None, charset=None):
        self.data = pd.read_csv(csv_path)
        self.img_height = img_height

        # Remove resizing to max_width here — we'll handle padding in collate_fn
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Build charset if not provided
        if charset is None:
            text = ''.join(self.data['label'].astype(str).values)
            self.charset = sorted(set(text))
        else:
            self.charset = charset

        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.charset)}  # CTC: 0 = blank
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label_str = str(row['label'])

        # Load image
        image = Image.open(image_path).convert('L')

        # Resize by height and maintain aspect ratio
        w, h = image.size
        new_w = int(self.img_height * w / h)
        image = image.resize((new_w, self.img_height), Image.Resampling.LANCZOS)

        image = self.transform(image)  # shape: [1, H, W]

        # Encode label
        label_encoded = torch.tensor(
            [self.char2idx[c] for c in label_str if c in self.char2idx],
            dtype=torch.long
        )
        #print(f"Label str: {label_str} | Encoded: {label_encoded} | Shape: {label_encoded.shape}")
        return {
            "image": image,               # Tensor [1, H, W]
            "label": label_encoded,       # Tensor [L]
            "label_str": label_str,       # For display
            "image_path": image_path      # Optional
        }

    def get_charset(self):
        return self.charset

    def get_mapping(self):
        return self.char2idx, self.idx2char
