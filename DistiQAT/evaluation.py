import evaluate
import torch
from dataset import SemanticSegmentationDataset
from transformers import SegformerImageProcessor
from SegFormer import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import json
from PIL import Image

import numpy as np
from ade20k_utils import ade_palette


class Metrics:
    @staticmethod
    def get_dataloader(root_dir, batch_size=8):
        processor = SegformerImageProcessor(reduce_labels=True)
        dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=processor, train=False)
        
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return dataloader

    @staticmethod
    def save_outputs(images, output, processor, batch_idx=0, batch_size=0):
        for idx in range(len(images)):
            predicted_segmentation_map = processor.post_process_semantic_segmentation(output[idx], target_sizes=[images[idx].shape[::-1]])[0]
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

            color_seg = np.zeros((predicted_segmentation_map.shape[0],
                predicted_segmentation_map.shape[1], 3), 
                dtype=np.uint8
            ) # height, width, 3

            palette = np.array(ade_palette())
            for label, color in enumerate(palette):
                color_seg[predicted_segmentation_map == label, :] = color

            # Convert to BGR
            color_seg = color_seg[..., ::-1]

            # Show image + mask
            img = np.array(images[idx]) * 0.5 + color_seg * 0.5
            img = img.astype(np.uint8)

            img = Image.fromarray(img)
            img.save(f"../data/ADEChallengeData2016/output/{batch_idx * batch_size + idx}.jpeg")

    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataset: str,
        save_output: bool = False,
    ):
        if save_output:
            processor = SegformerImageProcessor(do_resize=False)

        id2label = json.load(open('id2label.json', "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        result = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 8
        dataloader = Metrics.get_dataloader(dataset, batch_size=batch_size)
        metric = evaluate.load('mean_iou')
        model.to(device)

        model.eval()
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                output = model(images)
                logits = output.logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

            if save_output:
                Metrics.save_outputs(images=images.detach().cpu().numpy(), output=output, processor=processor, batch_idx=idx, batch_size=batch_size)

            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        result = metric._compute(
            predictions=predicted.cpu(),
            references=labels.cpu(),
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=False,
        )

        return result


if __name__ == "__main__":
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b4-finetuned-ade-512-512')
    results = Metrics.evaluate_model(model, dataset='../data/ADEChallengeData2016', save_output=True)
    print(results)

