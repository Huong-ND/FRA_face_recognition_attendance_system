"""
tools/enroll_deepfunneled.py
============================
Batch-enroll the LFW-deepfunneled dataset into the FRA database.

Usage (from FRA-main/ root):
    python tools/enroll_deepfunneled.py
    python tools/enroll_deepfunneled.py --limit 500   # only first 500 people
    python tools/enroll_deepfunneled.py --min-images 2

Pipeline per person:
  1. Iterate every image in their folder.
  2. Run SCRFD detection + ArcFace embedding on each image.
  3. If a face is detected, collect the embedding.
  4. Average all embeddings for that person → single representative vector.
  5. L2-normalise the average and INSERT INTO students.

Duplicate guard: if a student_code (= folder name) already exists, skip.
"""

import os
import sys
import argparse
import numpy as np

# Make sure FRA-main/ is on PYTHONPATH when run from tools/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from database.db import init_db
from models.student_model import create_student, get_student_by_code
from services.face_service import detect_and_embed, average_embeddings


def parse_args():
    p = argparse.ArgumentParser(description="Enroll LFW-deepfunneled into FRA DB")
    p.add_argument("--dataset", default=Config.DATASET_PATH,
                   help="Path to lfw-deepfunneled root folder")
    p.add_argument("--limit",   type=int, default=0,
                   help="Max number of people to enroll (0 = all)")
    p.add_argument("--min-images", type=int, default=1,
                   help="Skip people with fewer images than this")
    p.add_argument("--save-npz", action="store_true",
                   help="Also save gallery_lfw.npz after enrollment")
    return p.parse_args()


def enroll_person(person_name: str, folder_path: str) -> bool:
    """
    Extract embeddings from all images in folder, average them,
    and insert the student if not already present.
    Returns True on success.
    """
    # Skip if already enrolled
    code = person_name  # use folder name as student_code
    if get_student_by_code(code):
        return False  # already exists

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        return False

    embeddings = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = detect_and_embed(img)
        if faces:
            embeddings.append(faces[0]["embedding"])   # take first (best) face

    if not embeddings:
        return False

    avg_emb = average_embeddings(embeddings)
    display_name = person_name.replace("_", " ")   # "Aaron_Eckhart" → "Aaron Eckhart"
    create_student(display_name, code, avg_emb, source="lfw")
    return True


def save_npz(output_path: str):
    """Dump the gallery to .npz for fast offline loading."""
    from models.student_model import load_gallery
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    names, matrix = load_gallery()
    np.savez_compressed(output_path, names=np.array(names), embeddings=matrix)
    print(f"[npz] Saved {len(names)} embeddings → {output_path}")


def main():
    args = parse_args()
    dataset_root = args.dataset

    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset folder not found: {dataset_root}")
        sys.exit(1)

    print(f"[enroll] Dataset : {dataset_root}")
    print(f"[enroll] Initialising DB …")
    init_db()

    # Lazy-load insightface here so the model downloads only when needed
    from services import embedding_cache  # noqa – triggers model load

    person_folders = sorted(os.listdir(dataset_root))
    if args.limit:
        person_folders = person_folders[: args.limit]

    total   = len(person_folders)
    ok      = 0
    skipped = 0
    errors  = 0

    for idx, person_name in enumerate(person_folders, 1):
        folder = os.path.join(dataset_root, person_name)
        if not os.path.isdir(folder):
            continue

        n_imgs = len([f for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if n_imgs < args.min_images:
            skipped += 1
            continue

        try:
            result = enroll_person(person_name, folder)
            if result:
                ok += 1
                if ok % 50 == 0 or idx == total:
                    print(f"  [{idx}/{total}] enrolled={ok}  skipped={skipped}  errors={errors}")
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  [WARN] {person_name}: {e}")

    print(f"\n[enroll] Done. enrolled={ok}  skipped={skipped}  errors={errors}")

    # Refresh in-memory cache
    from services import embedding_cache
    embedding_cache.refresh()

    if args.save_npz:
        save_npz(Config.GALLERY_NPZ)


if __name__ == "__main__":
    main()
