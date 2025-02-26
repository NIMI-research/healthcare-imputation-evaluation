import os
import csv
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation:

    def scores(og_col, imputed_col, file_name, imputer, feature):
        mse = mean_squared_error(og_col, imputed_col)
        mae = mean_absolute_error(og_col, imputed_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(og_col, imputed_col)

        # Normalizing metrics using the range of true values
        y_min, y_max = min(og_col), max(og_col)
        range_of_values = y_max - y_min

        normalized_mse = mse / range_of_values
        normalized_mae = mae / range_of_values
        normalized_rmse = rmse / range_of_values

        file_path = "evals/evaluations.csv"
        flag = os.path.exists(file_path)

        time = datetime.now().strftime("%d/%m/%y %H:%M:%S")

        
        with open(file_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            if flag:
                pass
            else:
                dw = csv.DictWriter(
                    csv_file,
                    delimiter=",",
                    fieldnames=[
                        "File",
                        "Imputer",
                        "Time Stamp",
                        "Mean Squared Error (MSE)",
                        "Mean Absolute Error (MAE)",
                        "Root Mean Square Error (RMSE)",
                        "R2 Score",
                        "Feature"
                    ],
                )

                dw.writeheader()

            csv_writer.writerow([file_name, imputer, time, normalized_mse, normalized_mae, normalized_rmse, r2, feature])

        print(f"Evalution for {file_name} saved at {file_path}.")

    