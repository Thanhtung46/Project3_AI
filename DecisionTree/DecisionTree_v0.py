import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
import handledata as dt
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from train_script import grid

# Khởi tạo mô hình cây quyết định bình thường
baseline_model = grid.best_estimator_

# Huấn luyện mô hình trên tập dữ liệu đã tiền xử lý
baseline_model.fit(dt.X_train, dt.y_train) 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Khởi tạo khung hình
plt.figure(figsize=(33,20))

# Vẽ cây (giới hạn max_depth=3 để dễ nhìn, dù cây thật có thể sâu hơn)
plot_tree(baseline_model, 
          feature_names=dt.X.columns, 
          filled=True, 
          rounded=True, 
          fontsize=10,
          max_depth=3) 

plt.title("Decision Tree Structure - Baseline Model (Top 4 levels)")
plt.savefig('./Image/V0/Decision_Tree_Structure_v0.png')
# plt.show()


import seaborn as sns

# Lấy kết quả dự báo từ tập Valid
y_pred = baseline_model.predict(dt.X_valid)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(dt.y_valid, y_pred) 
"""Trung bình độ lệch tuyệt đối giữa dự đoán và giá trị thật"""
mse = mean_squared_error(dt.y_valid, y_pred)
"""Trung bình bình phương sai số"""
rmse = np.sqrt(mse) # Tính thêm RMSE vì Lab yêu cầu 

"""Căn bậc hai của MSE"""
# 1. R2 Test: Khả năng dự báo trên dữ liệu mới (y_pred đã có ở trên)
r2_test = r2_score(dt.y_valid, y_pred)

# 2. R2 Train: Khả năng học trên dữ liệu cũ
# model.score tự động dùng X_train để dự đoán rồi so sánh với y_train
r2_train = baseline_model.score(dt.X_train, dt.y_train)

# 3. Overfitting Gap: Độ lệch giữa học và hành
gap = r2_train - r2_test

from log import log_results
log_results("Model V0 - Baseline", mae, mse, rmse, r2_train, r2_test, gap)

plt.figure(figsize=(10, 6))

# Vẽ biểu đồ Scatter (Điểm)
sns.scatterplot(x=dt.y_valid, y=y_pred, alpha=0.5)

# Vẽ đường chéo chuẩn (Nếu dự báo đúng 100%, các điểm sẽ nằm trên đường này)
plt.plot([dt.y_valid.min(), dt.y_valid.max()], [dt.y_valid.min(), dt.y_valid.max()], '--r', linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Compare actual prices and predicted prices")
plt.savefig('./Image/V0/Actual_vs_Predicted_V0.png')
# plt.show()

# Lấy độ quan trọng của các đặc trưng
importances = baseline_model.feature_importances_
indices = np.argsort(importances)[:]  # Lấy top 10 đặc trưng quan trọng nhất

plt.figure(figsize=(10, 6))
plt.title('The most important factors affecting house prices')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [dt.X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('./Image/V0/Feature_Importance_V0.png')
# plt.show()
