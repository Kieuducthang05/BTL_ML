# -*- coding: utf-8 -*-
"""
File chuẩn hóa dữ liệu MABe Challenge - Phiên bản Final Complete
Chức năng:
1. Đọc dữ liệu Tracking (Parquet) và Metadata (CSV).
2. Chuẩn hóa Tọa độ (chia Width/Height), Vận tốc (chia PPC).
3. Chuẩn hóa Metadata tĩnh: Tuổi, Giới tính, Giống, Lồng, FPS.
4. Merge tất cả vào một file Parquet duy nhất cho mỗi Video.
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys
import io
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

# Fix encoding cho console (tránh lỗi hiển thị tiếng Việt trên Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = '../data'  # Thư mục gốc chứa data tải về
TRACKING_DIR = os.path.join(BASE_DIR, 'train_tracking')
META_PATH = os.path.join(BASE_DIR, 'train.csv')
OUTPUT_DIR = '../processed_data_normalized' # Thư mục lưu kết quả

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- PHẦN 1: XỬ LÝ METADATA (STATIC FEATURES) ---
def get_static_features_map(meta_path):
    """
    Đọc file train.csv và chuẩn hóa toàn bộ thông tin tĩnh.
    Output: Dictionary {video_id: {feature_name: value, ...}}
    """
    print(f"-> Đang xử lý Metadata từ {meta_path}...")
    
    # Đọc file, ép kiểu video_id về string để làm Key chuẩn
    df = pd.read_csv(meta_path, dtype={'video_id': str})
    
    # 1. Chuẩn hóa TUỔI (Age) - Code từ mousee.txt
    age_mapping = {
        np.nan: 0, '4-6 weeks': 1, '6-12 weeks': 2, 
        '8-12 weeks': 3, '12 weeks': 4, '>40 weeks': 5, '> 40 weeks': 5
    }
    df['mouse1_age_code'] = df['mouse1_age'].map(age_mapping).fillna(0).astype('int8')
    df['mouse2_age_code'] = df['mouse2_age'].map(age_mapping).fillna(0).astype('int8')

    # 2. Chuẩn hóa HÌNH DẠNG LỒNG (Arena Shape)
    shape_mapping = {'square': 0, 'rectangular': 1, 'circular': 2}
    df['arena_shape_code'] = df['arena_shape'].map(shape_mapping).fillna(-1).astype('int8')

    # 3. Chuẩn hóa LOẠI LỒNG (Arena Type)
    # Chuẩn hóa chuỗi về chữ thường trước khi map
    df['arena_type'] = df['arena_type'].str.lower().str.strip()
    type_mapping = {
        'familiar': 0, 'resident-intruder': 1, 
        'csds': 2, 'neutral': 3,
        np.nan: -1, '': -1
    }
    df['arena_type_code'] = df['arena_type'].map(type_mapping).fillna(-1).astype('int8')

    # 4. Chuẩn hóa GIỚI TÍNH (Sex) - QUAN TRỌNG
    sex_mapping = {'male': 1, 'female': 0}
    df['mouse1_sex_code'] = df['mouse1_sex'].map(sex_mapping).fillna(-1).astype('int8')
    df['mouse2_sex_code'] = df['mouse2_sex'].map(sex_mapping).fillna(-1).astype('int8')

    # 5. Chuẩn hóa GIỐNG LOÀI (Strain)
    # Tự động lấy danh sách các giống và đánh số
    strain_list = df['mouse1_strain'].unique()
    # Tạo dict: {'C57Bl/6': 0, 'CD-1': 1, ...}
    strain_mapping = {k: v for v, k in enumerate(strain_list)}
    
    df['mouse1_strain_code'] = df['mouse1_strain'].map(strain_mapping).fillna(-1).astype('int8')
    # Giả sử mouse2 cùng giống hoặc map tương tự (ở đây demo mouse1 đại diện)
    
    # 6. Chuẩn hóa FPS (Tốc độ khung hình)
    # Chia cho 30.0 để đưa về tỉ lệ so với chuẩn 30fps
    df['fps_norm'] = df['frames_per_second'] / 30.0

    # 7. Thông số kỹ thuật Video (cho hàm normalize)
    # Chúng ta gom luôn vào đây để tiện tra cứu 1 lần
    # Lưu ý: Cần xử lý tên Lab
    
    # Tạo Dictionary kết quả
    static_map = {}
    
    feature_cols = [
        'mouse1_age_code', 'mouse2_age_code',
        'arena_shape_code', 'arena_type_code',
        'mouse1_sex_code', 'mouse2_sex_code',
        'mouse1_strain_code', 'fps_norm',
        'lab_id', # Giữ lại để tạo folder
        'video_width_pix', 'video_height_pix', 'pix_per_cm_approx' # Giữ lại để tính toán
    ]

    # Convert DataFrame to Dict: {video_id: {col1: val1, col2: val2...}}
    static_map = df.set_index('video_id')[feature_cols].to_dict('index')
    
    return static_map, strain_mapping # Trả về strain_mapping để lưu lại nếu cần

# --- PHẦN 2: XỬ LÝ VÀ MERGE (MAIN LOGIC) ---
def normalize_and_merge_file(file_path, static_map):
    """
    Hàm xử lý cho 1 file Parquet (Tối ưu bản 2):
    1. Tìm Video ID chuẩn.
    2. Lấy thông số từ static_map.
    3. Tính toán Tracking (X, Y, Speed) - tối ưu memory.
    4. Ghép thêm cột Metadata.
    5. Lưu file.
    """
    filename = os.path.basename(file_path)
    
    # 1. Đọc file Tracking (dtype tối ưu)
    try:
        df = pd.read_parquet(file_path)
    except:
        return

    # 2. Xác định VIDEO ID
    if 'video_id' in df.columns:
        video_id = str(df['video_id'].iloc[0])
    else:
        video_id = filename.replace('.parquet', '')

    # 3. Tra cứu Metadata
    video_meta = static_map.get(video_id)
    if video_meta is None:
        return 

    # Lấy thông số
    width = video_meta['video_width_pix']
    height = video_meta['video_height_pix']
    ppc = max(video_meta['pix_per_cm_approx'], 0.01)  # Tránh chia cho 0
    lab_id = video_meta['lab_id']

    # 4. Pivot: Tạo part_id và pivot
    df['part_id'] = df['mouse_id'].astype(str) + '_' + df['bodypart']
    
    # Tối ưu: Dùng unstack thay vì pivot_table (nhanh hơn)
    df_pivot = df.set_index(['video_frame', 'part_id'])[['x', 'y']].unstack(fill_value=np.nan)
    
    # Lấy x và y riêng
    df_x = df_pivot['x']
    df_y = df_pivot['y']
    
    # 5. Nội suy nhanh (dùng interpolate trực tiếp)
    df_x.interpolate(method='linear', limit_direction='both', inplace=True)
    df_y.interpolate(method='linear', limit_direction='both', inplace=True)
    df_x.fillna(0, inplace=True)
    df_y.fillna(0, inplace=True)
    
    # Lưu bản gốc (dùng values để tiết kiệm memory)
    x_raw = df_x.values.astype(np.float32)
    y_raw = df_y.values.astype(np.float32)
    
    # 6. Chuẩn hóa vị trí (in-place để tiết kiệm memory)
    x_norm = (x_raw / width).astype(np.float32)
    y_norm = (y_raw / height).astype(np.float32)
    
    # 7. Tính vận tốc (tối ưu numpy)
    dx = np.diff(x_raw, axis=0, prepend=0)
    dy = np.diff(y_raw, axis=0, prepend=0)
    speed = np.sqrt(dx**2 + dy**2) / ppc
    speed = np.nan_to_num(speed, posinf=0, neginf=0).astype(np.float32)
    
    # 8. Tạo DataFrame kết quả (tiết kiệm memory)
    result = pd.DataFrame(index=df_x.index, dtype=np.float32)
    
    # Thêm cột X (float32 để tiết kiệm)
    for i, col in enumerate(df_x.columns):
        result[f'{col}_x'] = x_norm[:, i]
    
    # Thêm cột Y
    for i, col in enumerate(df_y.columns):
        result[f'{col}_y'] = y_norm[:, i]
    
    # Thêm cột Speed
    for i, col in enumerate(df_x.columns):
        result[f'{col}_speed'] = speed[:, i]
    
    # 9. Thêm metadata (int8 để tiết kiệm memory)
    cols_to_add = [
        'mouse1_age_code', 'mouse2_age_code',
        'arena_shape_code', 'arena_type_code',
        'mouse1_sex_code', 'mouse2_sex_code',
        'mouse1_strain_code'
    ]
    
    for col in cols_to_add:
        result[col] = np.int8(video_meta.get(col, 0))
    
    result['fps_norm'] = np.float32(video_meta.get('fps_norm', 1.0))

    # 10. Lưu file
    lab_dir = os.path.join(OUTPUT_DIR, str(lab_id))
    os.makedirs(lab_dir, exist_ok=True)
    
    save_name = filename.replace('.parquet', '_norm.parquet')
    save_path = os.path.join(lab_dir, save_name)
    
    result.to_parquet(save_path, index=True)

# --- MAIN BLOCK ---
if __name__ == "__main__":
    print("=== BẮT ĐẦU QUY TRÌNH CHUẨN HÓA DỮ LIỆU (FULL) ===")
    
    # 1. Tạo bản đồ Metadata
    static_map, strain_map = get_static_features_map(META_PATH)
    print(f"-> Đã load thông tin cho {len(static_map)} videos.")
    print(f"-> Danh sách giống loài (Strain Map): {strain_map}")
    
    # 2. Quét file Tracking
    # Tìm đệ quy trong thư mục con (**/*.parquet)
    all_files = glob.glob(os.path.join(TRACKING_DIR, '**', '*.parquet'), recursive=True)
    print(f"-> Tìm thấy {len(all_files)} file tracking cần xử lý.")
    
    # 3. Chạy xử lý (có thanh tiến trình tqdm)
    print("-> Bắt đầu xử lý từng file...")
    success_count = 0
    error_count = 0
    
    for file_path in tqdm(all_files, desc="Processing"):
        try:
            normalize_and_merge_file(file_path, static_map)
            success_count += 1
        except Exception as e:
            error_count += 1
            # In lỗi nhưng không dừng chương trình
            # print(f"Lỗi file {os.path.basename(file_path)}: {str(e)[:50]}")
            pass
            
    print(f"\n{'='*50}")
    print(f"✓ HOÀN TẤT QUY TRÌNH CHUẨN HÓA")
    print(f"{'='*50}")
    print(f"Thành công: {success_count}/{len(all_files)} file")
    if error_count > 0:
        print(f"Lỗi: {error_count} file")
    print(f"Dữ liệu được lưu tại: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*50}")