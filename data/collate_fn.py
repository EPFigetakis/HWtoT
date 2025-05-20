import torch
import torch.nn.functional as F

def ctc_collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    labels = [l.unsqueeze(0) if l.ndim == 0 else l for l in labels]

    # Pad images to same width
    max_width = max(img.shape[2] for img in images)
    padded_images = [
        F.pad(img, (0, max_width - img.shape[2]), mode='constant', value=0)
        for img in images
    ]
    images_tensor = torch.stack(padded_images)

    # Ensure all labels are 1D tensors
    label_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    

    # Estimate input lengths for CTC loss
    input_lengths = torch.full(
        size=(len(batch),),
        fill_value=(max_width // 4),  # adjust based on your CNN downsampling
        dtype=torch.long
    )

    return {
        "images": images_tensor,         # [B, 1, H, W]
        "label": labels_concat,         # 1D tensor of all label indices
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "label_strs": [b["label_str"] for b in batch],
        "image_paths": [b["image_path"] for b in batch]
    }