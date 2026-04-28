import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from log import log_results

# 1. Tải và Tiền xử lý dữ liệu trực tiếp (Cải tiến Missing Values)
data_path = os.path.join(project_root, 'data', 'data_set.csv')
df = pd.read_csv(data_path)
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
df['Car'] = df['Car'].fillna(0)
df = df.drop(columns=['Address', 'SellerG', 'Date', 'Postcode', 'Suburb', 'CouncilArea'])
df = pd.get_dummies(df, columns=['Type', 'Method', 'Regionname'])
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Huấn luyện với max_depth=5 để tăng độ chính xác
model_v3 = DecisionTreeRegressor(random_state=42, max_depth=5)
model_v3.fit(X_train, y_train)

# 3. Vẽ cấu trúc cây (Chỉ hiển thị đến tầng thứ 3 để hình đẹp)
plt.figure(figsize=(33,20))
plot_tree(model_v3, 
          feature_names=X.columns.tolist(), 
          filled=True, 
          rounded=True, 
          fontsize=10,
          max_depth=3) # <--- Chỉnh hiển thị lúc vẽ ở đây
plt.title("Decision Tree Structure - Improvement V3 (Trained depth=5, Plotted depth=3)")
plt.savefig('./Image/V3/Decision_Tree_Structure_v3.png')

# 4. Đánh giá (Tính toán dựa trên mô hình depth=5)
y_pred = model_v3.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)
log_results("Model V3 - Missing Values (Depth 5)", mae, mse, rmse, r2)

# 5. Vẽ biểu đồ Compare
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_valid, y=y_pred, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], '--r', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Compare Actual vs Predicted Prices - V3")
plt.savefig('./Image/V3/Actual_vs_Predicted_V3.png')

# 6. Vẽ biểu đồ Feature Importance
importances = model_v3.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.title('Important Factors Affecting House Prices - V3')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('./Image/V3/Feature_Importance_V3.png')

plt.show()