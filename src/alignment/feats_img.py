import argparse
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection, SiglipVisionModel
from safetensors.torch import save_file
from pathlib import Path
import json
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_root, annotation_root, image_processor):
        self.img_paths = []
        for file in Path(image_root).glob('*'):
            if file.is_file():
                self.img_paths.append(str(file))
        print(f"Found {len(self.img_paths)} images in {image_root}") 
        self.image_processor = image_processor
        self.annotation_root = Path(annotation_root)
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = Path(self.img_paths[idx])
        img = Image.open(path).convert('RGB')
        pixel_values = self.image_processor(images=img, return_tensors="pt")['pixel_values']

        with open(self.annotation_root / f"{path.stem}.json", 'r') as f:
            anns = json.load(f)
        assert next(iter(anns)) == path.name
        
        return {
            'id': path.stem,
            'pixel_values': pixel_values
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    if any(x in args.model_name_or_path.lower() for x in ['clip', 'openclip']):
        auto_cls = CLIPVisionModelWithProjection
    elif 'siglip' in args.model_name_or_path.lower() and not 'naflex' in args.model_name_or_path.lower():
        auto_cls = SiglipVisionModel
    else:
        auto_cls = AutoModel

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    output_path = f"{Path(args.output_path).joinpath(Path(args.model_name_or_path).name)}.safetensors"

    dataset = ImageDataset(args.image_root, args.annotation_root, image_processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,
        collate_fn=lambda x: x
    )

    dtype = torch.float32
    model = auto_cls.from_pretrained(args.model_name_or_path, device_map='cuda', dtype=dtype)
    
    all_embds = {}
    
    for _sample in tqdm(dataloader, desc="Extracting features"):
        sample = _sample[0]
        if sample is None:
            continue
        pixel_values = sample['pixel_values'].to(dtype=dtype, device=model.device)
        with torch.no_grad():
            outputs = model(pixel_values)
        
        all_embds[sample['id']] = outputs.pooler_output[0].cpu()
            
    save_file(all_embds, output_path)
    print(f"Saved {len(all_embds)} embeddings to {output_path}")