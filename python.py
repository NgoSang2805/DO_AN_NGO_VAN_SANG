import pandas as pd
import numpy as np

# Tạo ngẫu nhiên dữ liệu
np.random.seed(0)
num_patients = 200
ages = np.random.randint(40, 80, num_patients)
genders = np.random.choice(['Nam', 'Nữ'], num_patients)
family_histories = np.random.choice(['Có', 'Không'], num_patients)
white_blood_cells = np.random.randint(5000, 15000, num_patients)
blood_glucose_levels = np.round(np.random.uniform(4.5, 8.0, num_patients), 1)
test_results = np.random.choice(['Bình thường', 'Bất thường'], num_patients)
symptoms_list = ['Mệt mỏi', 'Giảm cân', 'Đau ngực', 'Khó thở', 'Ho kéo dài', 'Sốt', 'Chán ăn']
diagnoses = np.random.choice([0, 1], num_patients)

# Hàm tạo triệu chứng ngẫu nhiên
def generate_symptoms(symptoms_list):
    num_symptoms = np.random.randint(1, len(symptoms_list) + 1)
    return ', '.join(np.random.choice(symptoms_list, num_symptoms, replace=False))

# Tạo danh sách triệu chứng cho mỗi bệnh nhân
symptoms = [generate_symptoms(symptoms_list) for _ in range(num_patients)]

# Tạo dataframe
data = {
    'ID Bệnh nhân': range(1, num_patients + 1),
    'Tuổi': ages,
    'Giới tính': genders,
    'Tiền sử bệnh gia đình': family_histories,
    'Số lượng bạch cầu': white_blood_cells,
    'Mức độ đường huyết': blood_glucose_levels,
    'Kết quả xét nghiệm': test_results,
    'Triệu chứng': symptoms,
    'Chẩn đoán bệnh (0: Không, 1: Có)': diagnoses
}

df = pd.DataFrame(data)
print(df)

df.to_csv('output.csv', index=False, encoding='utf-8-sig')