import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Bước 1: Tải dữ liệu
df = pd.read_csv("student-mat.csv", sep=";")

# Bước 2: Tiền xử lý - Mã hóa biến định tính
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Bước 3: Tạo biến mục tiêu nhị phân: Đậu (1), Rớt (0)
df["pass"] = (df["G3"] >= 10).astype(int)

# Bước 4: Tách dữ liệu thành X (đặc trưng) và y (nhãn)
X = df.drop(columns=["G3", "pass"])  # Có thể giữ lại G1, G2 nếu muốn
y = df["pass"]

# Bước 5: Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bước 6: Huấn luyện mô hình Decision Tree với entropy (ID3)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Bước 7: Dự đoán và đánh giá
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Rớt", "Đậu"])

# Bước 8: Ghi kết quả ra file
with open("Ketqua.txt", "w", encoding="utf-8") as f:
    f.write("🎯 Accuracy: {:.4f}\n".format(acc))
    f.write("\n📊 Classification Report:\n")
    f.write(report)

# Bước 9: Vẽ và lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rớt", "Đậu"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - ID3")
plt.savefig("confusion_matrix.png")
plt.close()

# Bước 10: Vẽ và lưu cây quyết định
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["Rớt", "Đậu"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree - ID3")
plt.savefig("DecisionTree.png")
plt.close()

# Bước 11: Lưu mô hình
with open("student_pass_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Huấn luyện và lưu mô hình thành công!")
print("📄 Đã ghi kết quả vào 'Ketqua.txt'")
print("🖼️ Đã lưu ảnh 'confusion_matrix.png' và 'DecisionTree.png'")
