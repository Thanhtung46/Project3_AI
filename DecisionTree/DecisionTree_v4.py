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

# Cấu hình thư mục output V4
output_dir = os.path.join(project_root, "Image", "V4")
os.makedirs(output_dir, exist_ok=True)

# Cố định max_depth
FIXED_MAX_DEPTH = 5

# Tìm max_leaf_nodes tốt nhất trong giới hạn max depth
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, 
        max_depth=FIXED_MAX_DEPTH, 
        random_state=42
    )
    model.fit(X_train, y_train)
    preds_val = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds_val)

# Danh sách ứng viên (Dưới 32 lá vì depth=5 giới hạn 2^5 = 32)
candidate_max_leaf_nodes = [5, 10, 15, 20, 25, 30, 32]
y_train_fix = dt.X_train_y if hasattr(dt, 'X_train_y') else dt.y_train

scores = {leaf_size: get_mae(leaf_size, dt.X_train, dt.X_valid, y_train_fix, dt.y_valid) 
          for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# Huấn luyện mô hình V4 (Cải tiến kép: Depth + Leaf Nodes)
v4_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size, 
    max_depth=FIXED_MAX_DEPTH, 
    random_state=42
)
v4_model.fit(dt.X_train, y_train_fix)

# Tính toán các chỉ số
y_pred = v4_model.predict(dt.X_valid)
mae = mean_absolute_error(dt.y_valid, y_pred)
mse = mean_squared_error(dt.y_valid, y_pred)
rmse = np.sqrt(mse)

# Tính R2 và Gap
r2_train = v4_model.score(dt.X_train, y_train_fix)
r2_test = r2_score(dt.y_valid, y_pred)
overfitting_gap = r2_train - r2_test

print(f"--- KẾT QUẢ V4 (DEPTH={FIXED_MAX_DEPTH}, LEAF={best_tree_size}) ---")
print(f"R2 Train: {r2_train:.4f}")
print(f"R2 Test:  {r2_test:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.4f}")

# Log kết quả
log_results(f"Model V4 (Depth {FIXED_MAX_DEPTH} & Leaf {best_tree_size})", mae, mse, rmse, r2_train, r2_test, overfitting_gap)

# Lưu hình ảnh
# Vẽ cây
plt.figure(figsize=(20,10))
plot_tree(v4_model, feature_names=dt.X.columns, filled=True, rounded=True, fontsize=10)
plt.title(f"V4 Tree Structure (Gap: {overfitting_gap:.4f})")
plt.savefig(os.path.join(output_dir, 'Structure_V4.png'))

# Thực tế vs Dự đoán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dt.y_valid, y=y_pred, alpha=0.5, color='purple')
plt.plot([dt.y_valid.min(), dt.y_valid.max()], [dt.y_valid.min(), dt.y_valid.max()], '--r')
plt.title(f"Actual vs Predicted - V4 (R2 Test: {r2_test:.4f})")
plt.savefig(os.path.join(output_dir, 'Actual_vs_Predicted_V4.png'))

# Feature Importance
importances = v4_model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='plum')
plt.yticks(range(len(indices)), [dt.X.columns[i] for i in indices])
plt.title("Top 10 Feature Importances - V4")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Feature_Importance_V4.png'))

print("Đã hoàn thành file V4 và lưu kết quả!")