# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys
import io

# Fix encoding cho console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = '../data'
TRACKING_DIR = os.path.join(BASE_DIR, 'train_tracking')
META_PATH = os.path.join(BASE_DIR, 'train.csv')
OUTPUT_DIR = '../processed_data_normalized'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metadata_map(meta_path):
    """Đọc metadata và tạo dict tra cứu"""
    print(f"Đang đọc Metadata từ {meta_path}...")
    df = pd.read_csv(meta_path)
    df['video_id'] = df['video_id'].astype(str)
    
    meta_map = {}
    for _, row in df.iterrows():
        meta_map[row['video_id']] = {
            'width': row['video_width_pix'],
            'height': row['video_height_pix'],
            'ppc': row['pix_per_cm_approx'],
            'lab': row['lab_id']
        }
    return meta_map

def normalize_single_file(file_path, meta_map):
    """Xử lý 1 file - phiên bản tối ưu"""
    try:
        df = pd.read_parquet(file_path)
    except:
        return

    # Lấy video_id từ tên file
    video_id = os.path.basename(file_path).replace('.parquet', '')
    
    params = meta_map.get(video_id)
    if params is None:
        return
    
    width = params['width']
    height = params['height']
    ppc = params['ppc']
    lab_id = params['lab']

    # Chuyển sang dạng wide (frame x features)
    # Tạo ID duy nhất cho mỗi body part
    df['part_id'] = df['mouse_id'].astype(str) + '_' + df['bodypart']
    
    # Pivot nhanh hơn: từng tọa độ riêng
    df_x = df.pivot_table(index='video_frame', columns='part_id', values='x', aggfunc='first')
    df_y = df.pivot_table(index='video_frame', columns='part_id', values='y', aggfunc='first')
    
    # Nội suy (linear interpolation rất nhanh)
    df_x = df_x.interpolate(method='linear', limit_direction='both').fillna(0)
    df_y = df_y.interpolate(method='linear', limit_direction='both').fillna(0)
    
    # Lưu bản gốc để tính vận tốc
    df_x_raw = df_x.copy()
    df_y_raw = df_y.copy()
    
    # Chuẩn hóa vị trí
    df_x_norm = df_x / width
    df_y_norm = df_y / height
    
    # Chuẩn hóa tên cột
    df_x_norm.columns = [f'{col}_x' for col in df_x_norm.columns]
    df_y_norm.columns = [f'{col}_y' for col in df_y_norm.columns]
    
    # Tính vận tốc: (Euclidean distance) / ppc
    dx = df_x_raw.diff().fillna(0).values
    dy = df_y_raw.diff().fillna(0).values
    speed = np.sqrt(dx**2 + dy**2) / ppc
    
    df_speed = pd.DataFrame(speed, index=df_x_raw.index, 
                           columns=[f'{col}_speed' for col in df_x_raw.columns])
    
    # Ghép lại
    result = pd.concat([df_x_norm, df_y_norm, df_speed], axis=1)
    
    # Lưu file
    save_dir = os.path.join(OUTPUT_DIR, lab_id)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, os.path.basename(file_path).replace('.parquet', '_norm.parquet'))
    result.to_parquet(save_path, index=True)

if __name__ == "__main__":
    meta_map = load_metadata_map(META_PATH)
    
    all_files = glob.glob(os.path.join(TRACKING_DIR, '**', '*.parquet'), recursive=True)
    print(f"Tìm thấy {len(all_files)} file tracking.")
    
    print("Bắt đầu chuẩn hóa...")
    for file_path in tqdm(all_files):
        normalize_single_file(file_path, meta_map)
        
    print(f"Hoàn tất! Dữ liệu đã lưu tại {OUTPUT_DIR}")