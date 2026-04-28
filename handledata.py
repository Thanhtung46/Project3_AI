import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('./data/data_set.csv')

# 2. Loại bỏ cột thừa
features_to_drop = ['Address', 'SellerG', 'Date', 'Postcode']
df = df.drop(columns=features_to_drop)

# 3. Điền giá trị thiếu -> thay tất cả bằng 0
df = df.fillna(0)

# 4. Mã hóa (Encoding)
df = pd.get_dummies(df, columns=['Type', 'Method', 'Regionname'])

# 5. Tách X và y
X = df.drop(columns=['Price', 'Suburb', 'CouncilArea']) 
y = df['Price']

# 6. Chia tập dữ liệu 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)