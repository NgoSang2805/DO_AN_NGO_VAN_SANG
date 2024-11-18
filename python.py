import pandas as pd
import numpy as np

# Định nghĩa các giống mèo và số lượng
cats = {
    'Mèo Anh lông ngắn': 45,
    'Mèo Ba Tư': 55,
    'Mèo Scottish tai cụp': 43,
    'Mèo Ragdoll': 57
}

# Định nghĩa các thông số cho từng giống mèo
cat_data = {
    'Mèo Anh lông ngắn': {
        'weight': (4, 8.5),  # kg
        'height': (30, 46),  # cm
        'length': (50, 80),  # cm
        'age': (1, 20),      # years
        'colors': ['Màu xám', 'bicolor', 'tabby', 'lilac', 'silver', 'golden', 'xám xanh', 'trắng', 'đen']
    },
    'Mèo Ba Tư': {
        'weight': (2.5, 5),  # kg
        'height': (20, 30),  # cm
        'length': (40, 60),  # cm
        'age': (1, 15),      # years
        'colors': ['trắng', 'Creme', 'màu Hung', 'màu Lilac', 'màu xám xanh', 'màu socola', 'màu đen', 
                   'Creme và trắng', 'Hung và trắng', 'Lilac và trắng', 'xám xanh và trắng']
    },
    'Mèo Scottish tai cụp': {
        'weight': (3, 6.5),  # kg
        'height': (30, 50),  # cm
        'length': (40, 70),  # cm
        'age': (1, 16),      # years
        'colors': ['xanh đen', 'xanh xám', 'màu khói', 'vàng trắng', 'xám trắng', 'xám', 'đen']
    },
    'Mèo Ragdoll': {
        'weight': (3, 6.5),  # kg
        'height': (20, 35),  # cm
        'length': (40, 50),  # cm
        'age': (1, 16),      # years
        'colors': ['xanh lam', 'socola', 'hoa cà', 'đỏ', 'màu kem', 'Seal Lynx', 'Seal Tortie', 'xanh kem']
    }
}

# Tạo danh sách các mẫu dữ liệu cho mỗi giống mèo
data = []
for cat_breed, number in cats.items():
    for _ in range(number):
        cat_info = cat_data[cat_breed]
        weight = round(np.random.uniform(*cat_info['weight']), 1)
        height = round(np.random.uniform(*cat_info['height']))
        length = round(np.random.uniform(*cat_info['length']))
        age = np.random.randint(*cat_info['age'])
        color = np.random.choice(cat_info['colors'])
        data.append([cat_breed, color, height, length, weight, age])

# Tạo DataFrame
df = pd.DataFrame(data, columns=['Breed', 'Color', 'Height', 'Length', 'Weight', 'Age'])

# Xuất ra file CSV với mã hóa utf-8-sig
df.to_csv('cats_dataset.csv', index=False, encoding='utf-8-sig')

print(df.head())  # Hiển thị 5 hàng đầu tiên của dataset
