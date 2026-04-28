import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import handledata as dt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from log import log_results

# 1. Xác định các đặc trưng quan trọng từ mô hình ban đầu (Vẫn để cây mọc tự do để tìm Top 5 chuẩn nhất)
temp_model = DecisionTreeRegressor(random_state=42)
temp_model.fit(dt.X_train, dt.y_train)
importances = temp_model.feature_importances_
indices = np.argsort(importances)[-5:] # Lấy Top 5 quan trọng nhất
selected_features = [dt.X.columns[i] for i in indices]

# 2. Huấn luyện mô hình cải tiến với max_depth=5
X_train_fs = dt.X_train[selected_features]
X_valid_fs = dt.X_valid[selected_features]
model_v2 = DecisionTreeRegressor(random_state=42, max_depth=5)
model_v2.fit(X_train_fs, dt.y_train)

# 3. Vẽ cấu trúc cây (Chỉ hiển thị đến tầng thứ 3 để dễ đọc trong báo cáo)
plt.figure(figsize=(33,20))
plot_tree(model_v2, 
          feature_names=selected_features, 
          filled=True, 
          rounded=True, 
          fontsize=10,
          max_depth=3) # <--- Chỉ giới hạn hiển thị khi vẽ
plt.title("Decision Tree Structure - Improvement V2 (Trained depth=5, Plotted depth=3)")
plt.savefig('./Image/V2/Decision_Tree_Structure_v2.png')

# 4. Đánh giá và ghi Log (Dựa trên mô hình depth=5)
y_pred = model_v2.predict(X_valid_fs)
mae = mean_absolute_error(dt.y_valid, y_pred)
mse = mean_squared_error(dt.y_valid, y_pred)
rmse = np.sqrt(mse)

r2_test = r2_score(dt.y_valid, y_pred)
y_train_fix = dt.X_train_y if hasattr(dt, 'X_train_y') else dt.y_train
r2_train = model_v2.score(X_train_fs, y_train_fix)

gap = r2_train - r2_test
log_results("Model V2 - Feature Selection (Depth 5)", mae, mse, rmse, r2_train, r2_test, gap)

# 5. Vẽ biểu đồ Compare (Scatter)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dt.y_valid, y=y_pred, alpha=0.5)
plt.plot([dt.y_valid.min(), dt.y_valid.max()], [dt.y_valid.min(), dt.y_valid.max()], '--r', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Compare Actual vs Predicted Prices - V2 (Depth 5)")
plt.savefig('./Image/V2/Actual_vs_Predicted_V2.png')

# 6. Vẽ biểu đồ Feature Importance
feat_importances = model_v2.feature_importances_
feat_indices = np.argsort(feat_importances)
plt.figure(figsize=(10, 6))
plt.title('Important Factors Affecting House Prices - V2 (Depth 5)')
plt.barh(range(len(feat_indices)), feat_importances[feat_indices], color='b', align='center')
plt.yticks(range(len(feat_indices)), [selected_features[i] for i in feat_indices])
plt.xlabel('Relative Importance')
plt.savefig('./Image/V2/Feature_Importance_V2.png')

plt.show()