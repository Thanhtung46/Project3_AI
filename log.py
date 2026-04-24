import os
import pandas as pd
def log_results(model_name, mae, mse, rmse, r2, file_name = "./Compare_Models/model_comparison.csv"):
    results = {
        "Model Name" : [model_name],
        "MAE" : [mae],
        "MSE" : [mse],
        "RMSE" : [rmse],
        "R2_SCORE" : [r2]
    }
    df_new = pd.DataFrame(results)

    if not os.path.isfile(file_name):
        df_new.to_csv(file_name, index=False)
    else:
        df_new.to_csv(file_name, mode="a", header=False, index= False)
    print(f"save successful {model_name} at {file_name}")