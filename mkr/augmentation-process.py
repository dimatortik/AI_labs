import cv2
import yaml
import random
import numpy as np
from pathlib import Path
import shutil
import albumentations as A

SRC_ROOT = Path("spotify-dataset")
DST_ROOT = Path("spotify-augmented")

AUG_PER_IMAGE = 12
SEED = 42

OUT_SIZE = 640
MIN_MASK_AREA_PX = 40
MIN_VISIBILITY = 0.30
SIMPLIFY_EPS = 1.0
MAX_POLY_POINTS = 200

IMG_EXTS = (".jpg", ".jpeg", ".png")

random.seed(SEED)
np.random.seed(SEED)

for subset in ("train", "valid", "test"):
    (DST_ROOT / subset / "images").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / subset / "labels").mkdir(parents=True, exist_ok=True)


transform = A.Compose(
    [
        A.PadIfNeeded(min_height=OUT_SIZE, min_width=OUT_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        A.RandomCrop(height=OUT_SIZE, width=OUT_SIZE, p=1.0),

        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), shear=(-5, 5), p=0.7),
        A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.Blur(blur_limit=5, p=0.25),
    ],
)

# -------------------------
# YOLO-seg polygons
# -------------------------
def load_yolo_seg_polygons(txt_path: Path):
    """
    Returns: list[(cls:int, pts_norm: np.ndarray (N,2) float32)]
    YOLO-seg line: cls x1 y1 x2 y2 ...
    """
    if not txt_path.exists():
        return []

    txt = txt_path.read_text().strip()
    if not txt:
        return []

    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        nums = list(map(float, parts[1:]))
        if (len(nums) % 2) != 0:
            continue
        pts = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.float32)
        out.append((cls, pts))
    return out

def polygons_to_instance_mask(polys, w, h):
    """
    polys: list[(cls, pts_norm)]
    mask: uint16 0=bg, 1..K = instances
    cls_of_id: dict{instance_id: cls}
    orig_area: dict{instance_id: area_px}
    """
    mask = np.zeros((h, w), dtype=np.uint16)
    cls_of_id = {}
    orig_area = {}

    inst_id = 1
    for cls, pts_norm in polys:
        pts_px = np.stack([pts_norm[:, 0] * w, pts_norm[:, 1] * h], axis=1)
        pts_px = np.round(pts_px).astype(np.int32)
        pts_px[:, 0] = np.clip(pts_px[:, 0], 0, w - 1)
        pts_px[:, 1] = np.clip(pts_px[:, 1], 0, h - 1)

        if pts_px.shape[0] < 3:
            continue

        cv2.fillPoly(mask, [pts_px], color=inst_id)
        area = int((mask == inst_id).sum())
        if area <= 0:
            continue

        cls_of_id[inst_id] = cls
        orig_area[inst_id] = area
        inst_id += 1

    return mask, cls_of_id, orig_area

def instance_mask_to_polygons(mask, cls_of_id, orig_area, simplify_eps=0.0):
    """
    mask: uint16 instance mask
    returns: list[(cls, pts_px_float)]
    with filtering by MIN_VISIBILITY and MIN_MASK_AREA_PX
    """
    h, w = mask.shape[:2]
    out = []

    for inst_id, cls in cls_of_id.items():
        binm = (mask == inst_id).astype(np.uint8)
        new_area = int(binm.sum())
        if new_area < MIN_MASK_AREA_PX:
            continue

        oa = orig_area.get(inst_id, None)
        if oa is not None and oa > 0:
            if (new_area / oa) < MIN_VISIBILITY:
                continue

        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        if simplify_eps and simplify_eps > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon=simplify_eps, closed=True)

        pts = cnt.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            continue

        out.append((cls, pts))

    return out

def save_yolo_seg(txt_path: Path, polygons_px, w, h):
    """
    polygons_px: list[(cls, pts_px_float)]
    writes YOLO-seg polygon lines: cls x1 y1 x2 y2 ...
    """
    with open(txt_path, "w") as f:
        for cls, pts in polygons_px:
            if pts.shape[0] < 3:
                continue

            # limit polygon length
            if pts.shape[0] > MAX_POLY_POINTS:
                step = int(np.ceil(pts.shape[0] / MAX_POLY_POINTS))
                pts = pts[::step]
                if pts.shape[0] < 3:
                    continue

            xs = np.clip(pts[:, 0] / w, 0, 1)
            ys = np.clip(pts[:, 1] / h, 0, 1)

            coords = []
            for x, y in zip(xs, ys):
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")

            f.write(f"{int(cls)} " + " ".join(coords) + "\n")

# -------------------------
# PIPELINE
# -------------------------
def copy_subset_no_aug(subset):
    src_img_dir = SRC_ROOT / subset / "images"
    src_lbl_dir = SRC_ROOT / subset / "labels"
    dst_img_dir = DST_ROOT / subset / "images"
    dst_lbl_dir = DST_ROOT / subset / "labels"

    count = 0
    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        polys = load_yolo_seg_polygons(lbl_path)
        if not polys:
            continue
        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)
        count += 1

    print(f"{subset}: copied {count}")
    return count

def augment_train():
    src_img_dir = SRC_ROOT / "train" / "images"
    src_lbl_dir = SRC_ROOT / "train" / "labels"
    dst_img_dir = DST_ROOT / "train" / "images"
    dst_lbl_dir = DST_ROOT / "train" / "labels"

    saved = 0
    skipped = 0

    img_paths = [p for p in src_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    print(f"Found train images: {len(img_paths)}")

    for img_path in img_paths:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        polys = load_yolo_seg_polygons(lbl_path)
        if not polys:
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(str(dst_img_dir / img_path.name), bgr)
        shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)
        saved += 1

        inst_mask, cls_of_id, orig_area = polygons_to_instance_mask(polys, w, h)
        if len(cls_of_id) == 0:
            skipped += 1
            continue

        ok_aug = 0
        for i in range(AUG_PER_IMAGE):
            aug = transform(image=rgb, mask=inst_mask)
            aug_img = aug["image"]
            aug_mask = aug["mask"]

            polygons_px = instance_mask_to_polygons(
                aug_mask,
                cls_of_id=cls_of_id,
                orig_area=orig_area,
                simplify_eps=SIMPLIFY_EPS
            )
            if not polygons_px:
                continue

            aug_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(dst_img_dir / f"{aug_name}.jpg"), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            save_yolo_seg(dst_lbl_dir / f"{aug_name}.txt", polygons_px, w=OUT_SIZE, h=OUT_SIZE)
            ok_aug += 1
            saved += 1

        if ok_aug == 0:
            pass

    print(f"Train saved: {saved}, skipped: {skipped}")
    return saved

print("=== Build dataset ===")
valid_count = copy_subset_no_aug("valid")
test_count = copy_subset_no_aug("test")
train_count = augment_train()

yaml_dict = {
    "path": str(DST_ROOT.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["spotify_logo"],
}
with open(DST_ROOT / "data.yaml", "w") as f:
    yaml.dump(yaml_dict, f, default_flow_style=False)

print("DONE", {"train": train_count, "valid": valid_count, "test": test_count})
print("data.yaml ->", (DST_ROOT / "data.yaml").resolve())
