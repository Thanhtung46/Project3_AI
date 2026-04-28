from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import handledata as dt
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error'
)

grid.fit(dt.X_train, dt.y_train)