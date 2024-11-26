import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Đọc dữ liệu từ file CSV
df = pd.read_csv('cats_dataset.csv', encoding='utf-8-sig')

# Mã hóa biến phân loại
label_encoder = LabelEncoder()
df['Color'] = label_encoder.fit_transform(df['Color'])

# Xáo trộn và chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = df[['Color', 'Height', 'Length', 'Weight', 'Age']]
y = df['Breed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20/200, random_state=42, stratify=y, shuffle=True)

# Khởi tạo mô hình K-NN
knn = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# In ra kết quả dự đoán chi tiết từng con mèo trong tập kiểm tra
test_results = X_test.copy()
test_results['Actual Breed'] = y_test.values
test_results['Predicted Breed'] = y_pred

# Giải mã giá trị màu sắc trở lại tên gốc
test_results['Color'] = label_encoder.inverse_transform(test_results['Color'])

print(test_results)
