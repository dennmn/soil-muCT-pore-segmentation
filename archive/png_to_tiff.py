"""
PNG Sequence to 3D TIFF Converter
=================================
סקריפט זה מקבל תיקייה עם רצף תמונות PNG ומאחד אותן לקובץ TIFF תלת-ממדי אחד.
מתאים להכנת דאטה ל-nnUNet ולכלי האנוטציה שלו.
"""

import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from skimage import io

def create_3d_tiff_from_png_sequence(input_dir: Path, output_dir: Path, sample_name: str):
    """
    קורא רצף PNG, מאחד ל-Numpy Array ושומר כ-Multipage TIFF.
    """
    
    # בדיקה שהתיקייה קיימת
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return

    # 1. יצירת תיקיית פלט אם לא קיימת
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{sample_name}.tif"
    
    print(f"Checking input folder: {input_dir}")
    
    # 2. איסוף כל קבצי ה-PNG ומיון שלהם (חשוב מאוד לסדר ה-Z!)
    png_files = sorted(list(input_dir.glob("*.png")))
    
    if not png_files:
        print(f"❌ Error: No PNG files found in {input_dir}")
        return

    print(f"Found {len(png_files)} images. Starting stacking process...")

    # 3. קריאת התמונה הראשונה כדי לקבל את המימדים (Height, Width)
    first_img = io.imread(png_files[0])
    height, width = first_img.shape
    depth = len(png_files)
    
    # יצירת מערך ריק (Pre-allocation)
    volume_data = np.zeros((depth, height, width), dtype=first_img.dtype)

    # 4. קריאת התמונות והכנסה למערך
    for i, png_path in enumerate(tqdm(png_files, desc="Stacking Images")):
        img = io.imread(png_path)
        volume_data[i, :, :] = img

    print(f"Volume shape: {volume_data.shape}, Dtype: {volume_data.dtype}")

    # 5. שמירה כקובץ TIFF יחיד
    print(f"Saving 3D TIFF to: {output_file_path}")
    
    # דחיסת zlib מאוזנת בין מהירות לגודל קובץ
    tifffile.imwrite(output_file_path, volume_data, compression='zlib') 
    
    print("✅ Done! File is ready for annotation.")

# =========================
# CLI SETUP
# =========================
def main():
    parser = argparse.ArgumentParser(description="Convert a folder of PNG images to a single 3D TIFF file.")
    
    # הגדרת הארגומנטים שהסקריפט מקבל
    parser.add_argument(
        "-i", "--input", 
        type=Path, 
        required=True, 
        help="Path to the folder containing PNG images"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True, 
        help="Path to the output folder where the TIFF will be saved"
    )
    
    parser.add_argument(
        "-id", "--sample_id", 
        type=str, 
        required=True, 
        help="The name of the sample (will be the filename)"
    )

    args = parser.parse_args()

    # הפעלת הפונקציה עם הארגומנטים שהתקבלו
    create_3d_tiff_from_png_sequence(args.input, args.output, args.sample_id)

if __name__ == "__main__":
    main()