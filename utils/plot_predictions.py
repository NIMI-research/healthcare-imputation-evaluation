import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

def plot_line(predictions, actual, dataset_name: str, imputer: str, variable: str):
    # Create a DataFrame
    df = pd.DataFrame({'Actual': actual, 'Predictions': predictions})
    
    method = {
        "knn_5" : "KNN (k=5)",
        "knn_11" : "KNN (k=11)",
        "knn_15" : "KNN (k=15)",
        "mice" : "MICE",
        "miceforest" : "MICE Forest",
        "nn" : "GRU",
        "ssl" : "SSL"
    }

    # Plotting the data using Seaborn
    plt.figure(figsize=(17,10), dpi=500)
    ax = sns.lineplot(data=df, linewidth=2.5)
    ax.grid(False)
    
    # Customize the plot
    ax.set_title(f"Actual vs Predicted - {dataset_name} - {variable} - {method[imputer]}",fontsize=34, fontweight='bold')
    #ax.set(xlabel="Timesteps", ylabel="Observations")
    ax.set_xlabel('Timesteps', fontsize=34, fontweight='bold')
    ax.set_ylabel('Observations', fontsize=34, fontweight='bold')
    ax.tick_params(axis='x', labelsize=28) 
    ax.tick_params(axis='y', labelsize=28) 
    sns.despine()
    plt.legend()  # You can adjust the location as needed
    sns.move_legend(ax, "upper right", fontsize=24)
    plt.savefig(f"latest_imputation_plots/{dataset_name}_{imputer}_{variable}.png")
