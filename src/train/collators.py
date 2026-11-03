from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F


def pad_token_sequence(tokens: List[torch.Tensor], padding_value):
    return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=padding_value)


def _rescale_and_normalize(x, img_proc):
    """
    Apply rescale and normalize using the given image processor's parameters.
    Input x: numpy array (C, H, W) in [0, 255] range.
    Returns: numpy array (C, H, W) normalized.
    """
    x = img_proc.rescale(x, scale=img_proc.rescale_factor)
    # Some Fast image processors (e.g. DINOv3ViTImageProcessorFast) require a tensor input
    if type(img_proc).__name__ == 'DINOv3ViTImageProcessorFast' and not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    x = img_proc.normalize(x, mean=img_proc.image_mean, std=img_proc.image_std)
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x


class SupervisedCollator:
    def __init__(
        self,
        image_processor,
        tokenizer,
        tgt_image_processor=None,
        tgt_image_size: Optional[int] = None
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tgt_image_processor = tgt_image_processor
        self.tgt_image_size = tgt_image_size

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []
        images = []
        image_mask = []

        for sample in batch:
            input_ids.append(sample['input_ids'])
            attention_mask.append(sample['attention_mask'])
            labels.append(sample['labels'])
            images.append(sample['image'])
            image_mask.append(sample['image_mask'])

        input_ids = pad_token_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = pad_token_sequence(attention_mask, padding_value=0)
        labels = pad_token_sequence(labels, -100)
        image_mask = torch.tensor(image_mask, dtype=torch.bool)

        if self.tgt_image_processor is not None:
            # Step 1: shared spatial transformations (resize/crop) only, no rescale/normalize
            raw = self.image_processor(
                images, do_rescale=False, do_normalize=False, return_tensors='np'
            ).pixel_values  # (N, C, H, W), values in [0, 255]
            # Step 2: apply each processor's own rescale + normalize
            pixel_values = torch.from_numpy(
                np.stack([_rescale_and_normalize(raw[i], self.image_processor) for i in range(len(raw))], dtype=np.float32)
            )
            tgt_pixel_values = torch.from_numpy(
                np.stack([_rescale_and_normalize(raw[i], self.tgt_image_processor) for i in range(len(raw))], dtype=np.float32)
            )
        else:
            pixel_values = self.image_processor(images, return_tensors='pt').pixel_values
            tgt_pixel_values = None

        if self.tgt_image_size:
            tgt_pixel_values = F.interpolate(
                pixel_values if tgt_pixel_values is None else tgt_pixel_values,
                self.tgt_image_size,
                mode='bilinear',
                align_corners=False
            )

        ret = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_mask=image_mask
        )

        if tgt_pixel_values is not None:
            ret['tgt_pixel_values'] = tgt_pixel_values

        return ret
    