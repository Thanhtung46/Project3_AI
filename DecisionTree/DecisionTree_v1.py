import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Thêm đường dẫn để import handledata và log
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import handledata as dt
from log import log_results

# 1. Tạo thư mục output nếu chưa có
output_dir = os.path.join(project_root, "Image", "V1")
os.makedirs(output_dir, exist_ok=True)

# 2. Hàm để thử nghiệm các giá trị max_leaf_nodes khác nhau
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds_val)
    return mae

# Danh sách các giá trị cần thử nghiệm
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000]

# Tìm giá trị tốt nhất
scores = {leaf_size: get_mae(leaf_size, dt.X_train, dt.X_valid, dt.X_train_y if hasattr(dt, 'X_train_y') else dt.y_train, dt.y_valid) 
          for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

print(f"Giá trị max_leaf_nodes tối ưu nhất: {best_tree_size}")

# 3. Huấn luyện mô hình tối ưu nhất cho Cải tiến 1
improved_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=42)
improved_model.fit(dt.X_train, dt.y_train)

# 4. Đánh giá và lưu kết quả
y_pred = improved_model.predict(dt.X_valid)
mae = mean_absolute_error(dt.y_valid, y_pred)
mse = mean_squared_error(dt.y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(dt.y_valid, y_pred)

# Log kết quả vào file chung để so sánh với V0
log_results("Model V1 (Max Leaf Nodes)", mae, mse, rmse, r2)

# 5. Lưu hình ảnh vào thư mục V1
# Vẽ cấu trúc cây
plt.figure(figsize=(20,10))
plot_tree(improved_model, 
          feature_names=dt.X.columns, 
          filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title(f"Decision Tree Structure - V1 (max_leaf_nodes={best_tree_size})")
plt.savefig(os.path.join(output_dir, 'Decision_Tree_Structure_v1.png'))

# Vẽ biểu đồ thực tế vs dự đoán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dt.y_valid, y=y_pred, alpha=0.5)
plt.plot([dt.y_valid.min(), dt.y_valid.max()], [dt.y_valid.min(), dt.y_valid.max()], '--r')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted - V1 (MAE: {mae:,.0f})")
plt.savefig(os.path.join(output_dir, 'Actual_vs_Predicted_V1.png'))

print("Đã hoàn thành Cải tiến 1 và lưu ảnh vào Image/V1!")