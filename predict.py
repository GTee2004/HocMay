import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load model
with open("student_pass_model.pkl", "rb") as f:
    model = pickle.load(f)

# Danh sách các đặc trưng
features = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
    'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
    'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
]

# Mô tả và gợi ý nhập liệu bằng tiếng Việt
feature_labels = {
    'school': "Trường (GP/MS)",
    'sex': "Giới tính (M/F)",
    'age': "Tuổi (15-22)",
    'address': "Địa chỉ (U=thành phố, R=nông thôn)",
    'famsize': "Cỡ gia đình (GT3/LE3)",
    'Pstatus': "Tình trạng cha mẹ (T=cùng sống, A=không)",
    'Medu': "Trình độ học vấn mẹ (0-4)",
    'Fedu': "Trình độ học vấn cha (0-4)",
    'Mjob': "Nghề mẹ (teacher/health/services/at_home/other)",
    'Fjob': "Nghề cha (teacher/health/services/at_home/other)",
    'reason': "Lý do chọn trường (home/reputation/course/other)",
    'guardian': "Người giám hộ (mother/father/other)",
    'traveltime': "Thời gian đến trường (1-4)",
    'studytime': "Thời gian học mỗi tuần (1-4)",
    'failures': "Số môn trượt trước đó (0-3)",
    'schoolsup': "Học thêm ở trường (yes/no)",
    'famsup': "Học thêm gia đình hỗ trợ (yes/no)",
    'paid': "Học thêm trả phí (yes/no)",
    'activities': "Tham gia hoạt động ngoại khoá (yes/no)",
    'nursery': "Học mầm non (yes/no)",
    'higher': "Muốn học đại học (yes/no)",
    'internet': "Có internet ở nhà (yes/no)",
    'romantic': "Đang có mối quan hệ (yes/no)",
    'famrel': "Quan hệ gia đình (1=Rất tệ - 5=Rất tốt)",
    'freetime': "Thời gian rảnh (1-5)",
    'goout': "Thời gian đi chơi với bạn (1-5)",
    'Dalc': "Rượu ngày thường (1-5)",
    'Walc': "Rượu cuối tuần (1-5)",
    'health': "Tình trạng sức khoẻ (1-5)",
    'absences': "Số buổi nghỉ học",
    'G1': "Điểm kỳ 1 (0-20)",
    'G2': "Điểm kỳ 2 (0-20)"
}

# Mã hoá giá trị như mô hình
encoding_map = {
    "yes_no": {"yes": 1, "no": 0},
    "binary": {"T": 1, "A": 0, "U": 1, "R": 0, "GT3": 1, "LE3": 0},
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "jobs": {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4},
    "reason": {"home": 0, "reputation": 1, "course": 2, "other": 3},
    "guardian": {"mother": 0, "father": 1, "other": 2}
}

# Hàm xử lý input
def encode_input(values):
    try:
        data = []
        data.append(encoding_map["school"][values["school"]])
        data.append(encoding_map["sex"][values["sex"]])
        data.append(int(values["age"]))
        data.append(encoding_map["binary"][values["address"]])
        data.append(encoding_map["binary"][values["famsize"]])
        data.append(encoding_map["binary"][values["Pstatus"]])
        data.append(int(values["Medu"]))
        data.append(int(values["Fedu"]))
        data.append(encoding_map["jobs"][values["Mjob"]])
        data.append(encoding_map["jobs"][values["Fjob"]])
        data.append(encoding_map["reason"][values["reason"]])
        data.append(encoding_map["guardian"][values["guardian"]])
        data.append(int(values["traveltime"]))
        data.append(int(values["studytime"]))
        data.append(int(values["failures"]))
        data.append(encoding_map["yes_no"][values["schoolsup"]])
        data.append(encoding_map["yes_no"][values["famsup"]])
        data.append(encoding_map["yes_no"][values["paid"]])
        data.append(encoding_map["yes_no"][values["activities"]])
        data.append(encoding_map["yes_no"][values["nursery"]])
        data.append(encoding_map["yes_no"][values["higher"]])
        data.append(encoding_map["yes_no"][values["internet"]])
        data.append(encoding_map["yes_no"][values["romantic"]])
        data.append(int(values["famrel"]))
        data.append(int(values["freetime"]))
        data.append(int(values["goout"]))
        data.append(int(values["Dalc"]))
        data.append(int(values["Walc"]))
        data.append(int(values["health"]))
        data.append(int(values["absences"]))
        data.append(int(values["G1"]))
        data.append(int(values["G2"]))
        return np.array(data).reshape(1, -1)
    except Exception as e:
        messagebox.showerror("Lỗi dữ liệu", f"Dữ liệu không hợp lệ: {e}")
        return None

# Tạo giao diện
root = tk.Tk()
root.title("🎓 Dự đoán kết quả học tập (Đậu / Rớt)")

entries = {}

def predict():
    values = {key: entry.get() for key, entry in entries.items()}
    input_data = encode_input(values)
    if input_data is not None:
        prediction = model.predict(input_data)[0]
        result = "✅ ĐẬU" if prediction == 1 else "❌ RỚT"
        messagebox.showinfo("Kết quả dự đoán", f"Kết quả học tập dự đoán: {result}")

# Giao diện: Nhãn, ô nhập liệu, và chú thích
for i, feature in enumerate(features):
    tk.Label(root, text=feature).grid(row=i, column=0, sticky="w", padx=5)
    entry = tk.Entry(root, width=20)
    entry.grid(row=i, column=1)
    entries[feature] = entry
    # Gợi ý tiếng Việt
    tk.Label(root, text=feature_labels.get(feature, ""), fg="gray").grid(row=i, column=2, sticky="w", padx=5)

# Nút dự đoán
tk.Button(root, text="Dự đoán", command=predict).grid(row=len(features), column=0, columnspan=3, pady=10)

root.mainloop()
