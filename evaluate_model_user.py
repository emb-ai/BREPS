import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SAMMED2D_ALLOWED_PREFIXES = ("ACDC", "BUID", "MedScribble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified user-masks evaluation script for SAM-like models"
    )

    parser.add_argument("mode", nargs="?", default="NoBRS")

    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=[
            "sam",
            "robustsam",
            "mobile_sam",
            "sam2",
            "sammed2d",
            "samhq",
            "medsam",
        ],
    )

    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--print-ious", action="store_true")
    parser.add_argument("--save-ious", action="store_true")
    parser.add_argument("--datasets", type=str, default="FOR_TEST")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--iou-analysis", action="store_true")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--user_inputs", action="store_true")

    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--prompts_dir", type=str, default=None)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="real_user_data")
    parser.add_argument("--batch_size", type=int, default=16)

    # SAM-Med2D specific.
    # В старом рабочем запуске SAM-Med2D использовался image_size=256.
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--no_encoder_adapter",
        action="store_false",
        dest="encoder_adapter",
        help="Disable encoder_adapter for sammed2d",
    )
    parser.set_defaults(encoder_adapter=True)

    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.images_dir and args.prompts_dir:
        images_dir = args.images_dir
        prompts_dir = args.prompts_dir
        masks_dir = args.masks_dir or str(Path(images_dir).parent / "masks")
        return images_dir, prompts_dir, masks_dir

    dataset_name = args.datasets.split(",")[0]

    candidates = {
        "FOR_TEST": [
            (
                "datasets/FOR_TEST/images",
                "datasets/FOR_TEST/user_masks",
                "datasets/FOR_TEST/masks",
            ),
            (
                "FOR_TEST_resized/images",
                "FOR_TEST_resized/user_masks",
                "FOR_TEST_resized/masks",
            ),
        ]
    }

    if dataset_name not in candidates:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Pass --images_dir and --prompts_dir explicitly."
        )

    for images_dir, prompts_dir, masks_dir in candidates[dataset_name]:
        if os.path.isdir(images_dir) and os.path.isdir(prompts_dir):
            return images_dir, prompts_dir, masks_dir

    checked = "\n".join(
        [
            f"images={a}, prompts={b}, masks={c}"
            for a, b, c in candidates[dataset_name]
        ]
    )

    raise FileNotFoundError(
        f"Could not resolve dataset directories for {dataset_name}. Checked:\n{checked}"
    )


def get_prompts_by_image(images_dir: str, prompts_dir: str) -> Dict[str, List[Path]]:
    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    prompt_paths = {x.stem: x for x in Path(prompts_dir).glob("*.png")}

    samples_by_image: Dict[str, List[Path]] = {}

    for prompt_path in sorted(prompt_paths.values()):
        stem = prompt_path.stem
        image_name = stem[: stem.find("_1_")] if "_1_" in stem else stem

        if image_name == "BUID":
            image_name = "BUID_1"

        matched = [img for img in image_paths if image_name == Path(img).stem]

        if len(matched) != 1:
            raise AssertionError(
                f"Image match error for prompt {prompt_path}: {matched}"
            )

        image_path = matched[0]
        samples_by_image.setdefault(image_path, []).append(prompt_path)

    return samples_by_image


def read_mask_gray(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    return mask


def get_bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError("Empty prompt mask encountered.")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # SAM-style box format: x_min, y_min, x_max, y_max
    return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)


def calculate_iou(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    pred = pred_masks.astype(bool).reshape(pred_masks.shape[0], -1)
    gt = gt_masks.astype(bool).reshape(gt_masks.shape[0], -1)

    inter = np.logical_and(pred, gt).sum(axis=1).astype(np.float32)
    union = np.logical_or(pred, gt).sum(axis=1).astype(np.float32)

    return inter / (union + eps)


def batch_iou_torch(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if gt_masks.ndim == 4 and gt_masks.size(1) == 1:
        gt = gt_masks.squeeze(1).bool()
    else:
        gt = gt_masks.bool()

    if pred_masks.ndim == 4 and pred_masks.size(1) == 1:
        pred = pred_masks.squeeze(1).bool()
    else:
        pred = pred_masks.bool()

    gt_flat = gt.flatten(1)
    pred_flat = pred.flatten(1)

    inter = (gt_flat & pred_flat).sum(dim=1).float()
    union = (gt_flat | pred_flat).sum(dim=1).float()

    return inter / (union + eps)


def load_standard_predictor(
    family: str,
    model_type: str,
    checkpoint: str,
    device: str,
):
    if family == "sam":
        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything.utils.transforms import ResizeLongestSide

    elif family in {"robustsam", "mobile_sam"}:
        from custom_builds.MobileSAM.mobile_sam import sam_model_registry, SamPredictor
        from custom_builds.MobileSAM.mobile_sam.utils.transforms import ResizeLongestSide

    elif family == "medsam":
        from medsam.build_sam import sam_model_registry
        from medsam.predictor import SamPredictor
        from medsam.utils.transforms import ResizeLongestSide

    else:
        raise ValueError(f"Unsupported standard family: {family}")

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()

    predictor = SamPredictor(model)

    if family == "medsam":
        target_length = 1024
    else:
        target_length = getattr(model.image_encoder, "img_size", 1024)

    resize = ResizeLongestSide(target_length)

    return predictor, resize


def load_sammed2d_predictor(
    model_type: str,
    checkpoint: str,
    device: str,
    image_size: int = 256,
    encoder_adapter: bool = True,
):
    from sammed2d.build_sam import sam_model_registry
    from sammed2d.utils.transforms import ResizeLongestSide
    from sammed2d.predictor_sammed import SammedPredictor

    class SamMedArgs:
        pass

    model_args = SamMedArgs()
    model_args.image_size = image_size
    model_args.encoder_adapter = encoder_adapter
    model_args.sam_checkpoint = checkpoint

    model = sam_model_registry[model_type](model_args)
    model.to(device=device)
    model.eval()

    predictor = SammedPredictor(model)
    resize = ResizeLongestSide(image_size)

    return predictor, resize


@torch.inference_mode()
def evaluate_standard_family(
    args: argparse.Namespace,
    images_dir: str,
    prompts_dir: str,
    masks_dir: str,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, resize = load_standard_predictor(
        family=args.family,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=device,
    )

    dataset_samples = get_prompts_by_image(images_dir, prompts_dir)
    all_rows = []

    for img_path, prompt_paths in tqdm(
        dataset_samples.items(),
        desc=f"{args.family}:{args.model_type}",
    ):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask_path = str(Path(masks_dir) / f"{Path(img_path).stem}.png")
        gt_mask = read_mask_gray(gt_mask_path)

        gt_mask_t = resize.apply_image_torch(
            torch.tensor(gt_mask, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gt_mask_t = (gt_mask_t > 0).float()

        current_paths = [Path(gt_mask_path)] + list(prompt_paths)

        resized_image = resize.apply_image(image)
        predictor.set_image(resized_image)

        boxes = []
        for mask_path in current_paths:
            prompt_mask = read_mask_gray(str(mask_path))

            prompt_mask_t = resize.apply_image_torch(
                torch.tensor(prompt_mask, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            prompt_mask_t = (prompt_mask_t > 0).float()

            bbox = get_bbox_from_mask(prompt_mask_t.cpu().numpy()[0, 0])
            boxes.append(bbox)

        input_boxes = torch.tensor(np.stack(boxes), device=device)
        input_boxes = resize.apply_boxes_torch(
            input_boxes,
            resized_image.shape[:2],
        )

        pred_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )

        gt_stack = torch.stack([gt_mask_t] * len(current_paths)).to(device)

        pred_np = pred_masks.detach().cpu().numpy().squeeze(1) > args.thresh
        gt_np = gt_stack.detach().cpu().numpy().squeeze(1) > 0

        ious = calculate_iou(pred_np, gt_np)

        for idx, mask_path in enumerate(current_paths):
            all_rows.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "iou": float(ious[idx]),
                    "bbox": input_boxes[idx].detach().cpu().tolist(),
                    "family": args.family,
                    "model_type": args.model_type,
                    "checkpoint": args.checkpoint,
                }
            )

    return pd.DataFrame(all_rows)


@torch.inference_mode()
def evaluate_sammed2d_family(
    args: argparse.Namespace,
    images_dir: str,
    prompts_dir: str,
    masks_dir: str,
) -> pd.DataFrame:
    """
    SAM-Med2D evaluation path is intentionally close to the old standalone
    SAM-Med2D script:
      - image_size = 256
      - encoder_adapter = True
      - ResizeLongestSide(256)
      - IoU is computed by batch_iou_torch(gt_stack, pred_masks)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, resize = load_sammed2d_predictor(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=device,
        image_size=256,
        encoder_adapter=True,
    )

    dataset_samples = get_prompts_by_image(images_dir, prompts_dir)

    dataset_samples = {
        img_path: prompt_paths
        for img_path, prompt_paths in dataset_samples.items()
        if any(prefix in Path(img_path).stem for prefix in SAMMED2D_ALLOWED_PREFIXES)
    }

    all_rows = []

    for img_path, prompt_paths in tqdm(
        dataset_samples.items(),
        desc=f"sammed2d:{args.model_type}",
    ):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask_path = str(Path(masks_dir) / f"{Path(img_path).stem}.png")
        gt_mask = read_mask_gray(gt_mask_path)

        gt_mask_t = resize.apply_image_torch(
            torch.Tensor(gt_mask).unsqueeze(0).unsqueeze(0)
        )
        gt_mask_t = (gt_mask_t > 0).float()

        current_paths = [Path(gt_mask_path)] + list(prompt_paths)

        resized_image = resize.apply_image(image)
        predictor.set_image(resized_image, image_format="RGB")

        boxes = []
        for mask_path in current_paths:
            prompt_mask = read_mask_gray(str(mask_path))

            prompt_mask_t = resize.apply_image_torch(
                torch.Tensor(prompt_mask).unsqueeze(0).unsqueeze(0)
            )
            prompt_mask_t = (prompt_mask_t > 0).float()

            bbox = get_bbox_from_mask(prompt_mask_t.cpu().numpy()[0, 0])
            boxes.append(bbox)

        input_boxes = torch.tensor(np.stack(boxes), device=device)
        input_boxes = resize.apply_boxes_torch(
            input_boxes,
            resized_image.shape[:2],
        )

        pred_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )

        gt_stack = torch.stack([gt_mask_t] * len(current_paths)).to(device)

        ious = batch_iou_torch(gt_stack, pred_masks)

        for idx, mask_path in enumerate(current_paths):
            all_rows.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "iou": float(ious[idx].detach().cpu()),
                    "bbox": input_boxes[idx].detach().cpu().tolist(),
                    "family": "sammed2d",
                    "model_type": args.model_type,
                    "checkpoint": args.checkpoint,
                }
            )

    return pd.DataFrame(all_rows)


def sam2_model_cfg(model_type: str) -> str:
    mapping = {
        "t": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "s": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "b": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "l": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    if model_type not in mapping:
        raise ValueError(f"Unsupported SAM2 model_type: {model_type}")

    return mapping[model_type]


def set_image_as_batch_sam2(predictor, image: np.ndarray, batch_size: int):
    predictor.set_image(image)

    single_embed = predictor._features["image_embed"]
    single_high_res = predictor._features["high_res_feats"]

    predictor._features = {
        "image_embed": single_embed.repeat(batch_size, 1, 1, 1),
        "high_res_feats": [
            feat.repeat(batch_size, 1, 1, 1) for feat in single_high_res
        ],
    }

    predictor._is_image_set = True
    predictor._is_batch = True
    predictor._orig_hw = [predictor._orig_hw[0]] * batch_size

    return predictor


@torch.inference_mode()
def evaluate_sam2_family(
    args: argparse.Namespace,
    images_dir: str,
    prompts_dir: str,
    masks_dir: str,
) -> pd.DataFrame:
    from custom_builds.sam2.sam2.build_sam import build_sam2
    from custom_builds.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = sam2_model_cfg(args.model_type)
    sam2_model = build_sam2(model_cfg, args.checkpoint, device=device)
    sam2_model.eval()

    predictor = SAM2ImagePredictor(sam2_model)
    dataset_samples = get_prompts_by_image(images_dir, prompts_dir)

    all_rows = []

    for img_path, prompt_paths in tqdm(
        dataset_samples.items(),
        desc=f"sam2:{args.model_type}",
    ):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask_path = str(Path(masks_dir) / f"{Path(img_path).stem}.png")
        gt_mask = read_mask_gray(gt_mask_path)

        current_paths = [Path(gt_mask_path)] + list(prompt_paths)

        all_pred_masks = []
        all_boxes = []

        for start in range(0, len(current_paths), args.batch_size):
            batch_paths = current_paths[start: start + args.batch_size]

            predictor = set_image_as_batch_sam2(
                predictor,
                image,
                len(batch_paths),
            )

            batch_boxes = []
            for mask_path in batch_paths:
                prompt_mask = read_mask_gray(str(mask_path))
                bbox = get_bbox_from_mask(prompt_mask)
                batch_boxes.append(bbox)

            batch_boxes_np = np.stack(batch_boxes)

            pred_masks, _, _ = predictor.predict_batch(
                None,
                None,
                box_batch=batch_boxes_np,
                multimask_output=False,
            )

            predictor.reset_predictor()

            all_pred_masks.append(pred_masks.astype(bool))
            all_boxes.append(batch_boxes_np)

        pred_np = np.concatenate(all_pred_masks, axis=0)
        box_np = np.concatenate(all_boxes, axis=0)
        gt_np = np.stack([gt_mask > 0] * len(current_paths), axis=0)

        ious = calculate_iou(pred_np, gt_np)

        for idx, mask_path in enumerate(current_paths):
            all_rows.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "iou": float(ious[idx]),
                    "bbox": box_np[idx].tolist(),
                    "family": args.family,
                    "model_type": args.model_type,
                    "checkpoint": args.checkpoint,
                }
            )

    return pd.DataFrame(all_rows)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images_dir, prompts_dir, masks_dir = resolve_dataset_paths(args)

    if args.family == "sam2":
        df = evaluate_sam2_family(args, images_dir, prompts_dir, masks_dir)

    elif args.family == "sammed2d":
        df = evaluate_sammed2d_family(args, images_dir, prompts_dir, masks_dir)

    else:
        df = evaluate_standard_family(args, images_dir, prompts_dir, masks_dir)

    checkpoint_stem = Path(args.checkpoint).stem
    output_name = f"user_eval_{args.family}_{args.model_type}_{checkpoint_stem}.csv"
    output_path = Path(args.output_dir) / output_name

    if args.save_ious:
        df.to_csv(output_path, index=False)

    if args.print_ious:
        if len(df) == 0:
            print("No samples were evaluated.")
            print("Check dataset paths and SAMMED2D_ALLOWED_PREFIXES.")
            return

        print(df[["image_path", "mask_path", "iou"]].to_string(index=False))
        print("\nMean IoU:", float(df["iou"].mean()))
        print("Median IoU:", float(df["iou"].median()))
        print("Saved to:", output_path if args.save_ious else "not saved")


if __name__ == "__main__":
    main()