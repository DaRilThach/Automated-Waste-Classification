import os
import sys
import shutil
import random
from pathlib import Path
from PIL import Image
import kagglehub

# Thiết lập UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    CLASSES,
    RAW_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)

def locate_or_download_dataset():
    """Xac dinh hoac tai ve tap du lieu Kaggle Garbage Classification."""
    print("=" * 60)
    print("[*] BUOC 1: KIEM TRA & TAI TAP DU LIEU GARBAGE CLASSIFICATION")
    print("=" * 60)
    
    # 1. Kiem tra cache mac dinh cua kagglehub
    user_home = Path.home()
    possible_cache = user_home / ".cache" / "kagglehub" / "datasets" / "asdasdasasdas" / "garbage-classification" / "versions" / "2" / "Garbage classification" / "Garbage classification"
    
    if possible_cache.exists() and len(os.listdir(possible_cache)) >= 6:
        print(f"[+] Da tim thay du lieu tai: {possible_cache}")
        return possible_cache
        
    # 2. Neu chua co, tai tu dong qua kagglehub
    print("[*] Dang tai du lieu tu Kaggle Hub...")
    download_path = kagglehub.dataset_download('asdasdasasdas/garbage-classification')
    
    # Tim thu muc con chua cac thu muc lop
    candidate = Path(download_path) / "Garbage classification" / "Garbage classification"
    if candidate.exists():
        return candidate
    candidate2 = Path(download_path) / "Garbage classification"
    if candidate2.exists():
        return candidate2
    return Path(download_path)

def verify_and_split_dataset(source_dir: Path):
    """
    Xac thuc tinh toan ven cua anh va phan chia tap du lieu thanh Train/Val/Test
    theo phuong phap phan tang (Stratified).
    """
    print("\n" + "=" * 60)
    print("[*] BUOC 2: TIEN XU LY & PHAN CHIA TAP TRAIN / VAL / TEST")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    
    # Tao cau truc thu muc cho tung tap va tung lop
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            os.makedirs(split_dir / cls, exist_ok=True)
            
    summary_stats = []
    total_images_all = 0
    
    for cls in CLASSES:
        cls_src = source_dir / cls
        if not cls_src.exists():
            print(f"[!] Canh bao: Khong tim thay thu muc nhan '{cls}' tai {cls_src}")
            continue
            
        # Loc danh sach file anh hop le
        valid_files = []
        for filename in os.listdir(cls_src):
            filepath = cls_src / filename
            if filepath.is_file() and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_files.append(filepath)
                except Exception as e:
                    print(f"[-] File hong bo qua: {filepath} ({e})")
                    
        random.shuffle(valid_files)
        total_cls = len(valid_files)
        total_images_all += total_cls
        
        n_train = int(total_cls * TRAIN_RATIO)
        n_val = int(total_cls * VAL_RATIO)
        
        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]
        
        # Sao chep vao cac tap tuong ung
        for f in train_files:
            shutil.copy2(f, TRAIN_DIR / cls / f.name)
        for f in val_files:
            shutil.copy2(f, VAL_DIR / cls / f.name)
        for f in test_files:
            shutil.copy2(f, TEST_DIR / cls / f.name)
            
        summary_stats.append({
            "class": cls,
            "total": total_cls,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        })
        
    print(f"\n[+] BANG PHAN BO DU LIEU DU AN (Tong cong: {total_images_all} anh):")
    print("-" * 65)
    print(f"{'Lop Rac Thai':<15} | {'Tong so':<8} | {'Train (80%)':<12} | {'Val (10%)':<10} | {'Test (10%)':<10}")
    print("-" * 65)
    for stat in summary_stats:
        print(f"{stat['class']:<15} | {stat['total']:<8} | {stat['train']:<12} | {stat['val']:<10} | {stat['test']:<10}")
    print("-" * 65)
    print(f"{'TONG CONG':<15} | {total_images_all:<8} | "
          f"{sum(s['train'] for s in summary_stats):<12} | "
          f"{sum(s['val'] for s in summary_stats):<10} | "
          f"{sum(s['test'] for s in summary_stats):<10}")
    print("=" * 65)
    print("[+] Du lieu da san sang cho huan luyen va kiem thu!")

if __name__ == "__main__":
    source_dir = locate_or_download_dataset()
    verify_and_split_dataset(source_dir)
