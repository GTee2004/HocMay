import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Tắt cảnh báo Precision không xác định
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Biến toàn cục
model = None
feature_names = None
target_name = 'G3'

# Nạp dữ liệu từ file CSV hoặc Excel
def Load_Data(filepath):
    ext = filepath.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(filepath, sep=';')  # CSV phân cách bằng dấu ;
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Chỉ hỗ trợ định dạng .csv, .xls, .xlsx")
    
    df.columns = df.columns.str.strip()  # Xóa khoảng trắng trong tên cột
    return df

# Chuyển nhãn G3 thành nhị phân: G3 >= 10 là đậu (1), ngược lại là rớt (0)
def Convert_Label(data):
    data = data.copy()
    data[target_name] = data[target_name].apply(lambda x: 1 if x >= 10 else 0)
    return data

# Huấn luyện cây quyết định
def DecisionTree(data):
    global model, feature_names
    data = Convert_Label(data)

    X = data.drop(columns=[target_name])
    y = data[target_name]
    X = pd.get_dummies(X)  # Mã hóa biến phân loại
    feature_names = X.columns

    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X, y)

    print("\n=== Cây quyết định (text) ===")
    print(export_text(model, feature_names=list(feature_names)))

    # Vẽ và lưu cây
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Rớt", "Đậu"], filled=True, rounded=True)
    plt.title("Cây Quyết Định Dự Đoán Kết Quả Học Tập")
    plt.savefig("decision_tree.png")
    plt.show()

# Kiểm thử mô hình trên tập test
def Testing(data):
    global model, feature_names
    data = Convert_Label(data)

    X_test = data.drop(columns=[target_name])
    y_test = data[target_name]
    X_test = pd.get_dummies(X_test)

    # Đảm bảo cột trùng khớp với cột huấn luyện
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    y_pred = model.predict(X_test)

    # Lưu kết quả ra CSV
    results = data.copy()
    results['Dự đoán'] = y_pred
    results['Thực tế'] = y_test
    results.to_csv("prediction_results.csv", index=False)

    return y_test, y_pred

# Hiển thị hiệu suất mô hình
def Performance(y_true, y_pred):
    print("\n=== Đánh giá mô hình ===")
    print("Độ chính xác:", accuracy_score(y_true, y_pred))
    print("\nMa trận nhầm lẫn:\n", confusion_matrix(y_true, y_pred))
    print("\nBáo cáo phân loại:\n", classification_report(y_true, y_pred, zero_division=0))

    # Vẽ và lưu ma trận nhầm lẫn
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                                  display_labels=["Rớt", "Đậu"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Ma Trận Nhầm Lẫn")
    plt.savefig("confusion_matrix.png")
    plt.show()

# Hàm chính
if __name__ == "__main__":
    # Bước 1: Huấn luyện với student-mat.csv
    data_train = Load_Data("student-mat.csv")
    print("Tên cột trong dữ liệu huấn luyện:", data_train.columns.tolist())
    DecisionTree(data_train)

    # Bước 2: Kiểm thử với student-por.csv
    data_test = Load_Data("student-por.csv")
    y_true, y_pred = Testing(data_test)

    # Bước 3: Đánh giá hiệu quả
    Performance(y_true, y_pred)
