import os
import pandas as pd
def log_results(model_name, mae, mse, rmse, train, test, gap, file_name = "./Compare_Models/model_comparison.csv"):
    results = {
        "Model Name" : [model_name],
        "MAE" : [round(mae, 5)],
        "MSE" : [round(mse, 5)],
        "RMSE" : [round(rmse, 5)],
        "R2 Train" : [round(train, 5)],
        "R2 Test" : [round(test, 5)],
        "Evaluate" : [round(gap, 5)]
    }
    df_new = pd.DataFrame(results)

    if not os.path.isfile(file_name):
        df_new.to_csv(file_name, index=False)
    else:
        df_new.to_csv(file_name, mode="a", header=False, index= False)
    print(f"save successful {model_name} at {file_name}")