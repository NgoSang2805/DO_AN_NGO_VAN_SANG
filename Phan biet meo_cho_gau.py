import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Hàm để đọc ảnh và gán nhãn
def load_images_from_folder(folder, label, size=(64, 64)):
    images = []
    labels = []
    if not os.path.exists(folder):
        print("Thư mục không tồn tại:", folder)
        return images, labels
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("Không thể đọc file:", img_path)
            continue
        img = cv2.resize(img, size)
        images.append(img)
        labels.append(label)
    return images, labels

# Đường dẫn tới các thư mục chứa ảnh (đường dẫn tương đối)
cat_folder = os.path.join('dataset', 'cats')
dog_folder = os.path.join('dataset', 'dogs')
panda_folder = os.path.join('dataset', 'pandas')

# Load ảnh và gán nhãn
cat_images, cat_labels = load_images_from_folder(cat_folder, 'cat')
dog_images, dog_labels = load_images_from_folder(dog_folder, 'dog')
panda_images, panda_labels = load_images_from_folder(panda_folder, 'panda')

# Tạo dataset
X = np.array(cat_images + dog_images + panda_images)
y = np.array(cat_labels + dog_labels + panda_labels)

# Chuyển đổi dữ liệu và nhãn thành dạng mà model có thể xử lý
X = X.reshape(X.shape[0], -1)  # Chuyển đổi ảnh thành vector
y = np.array(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Khởi tạo mô hình K-NN
knn = KNeighborsClassifier(n_neighbors=11)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
