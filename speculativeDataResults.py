import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# traditionalMachineLearningF1 = pd.read_csv("Results/traditionalMethodsf1.csv")
# autoRegressiveModelF1 = pd.read_csv("Results/autoregressiveMethodsf1.csv")
traditionalMachineLearningF1 = pd.read_csv("Results/covidSpeculativef1.csv")
autoRegressiveModelF1 = pd.read_csv("Results/covidAutoRegressiveMethodsf1.csv")

f1_results = pd.concat([traditionalMachineLearningF1, autoRegressiveModelF1])
f1_results.reset_index(drop=True, inplace=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot
# sns.barplot(x="method", y="f1_score", hue="type", palette="bright", data=f1_results)
# plt.title("F1 Score by Method and Type")
plt.title("F1 Score by Method")
sns.barplot(x="method", y="f1_score", palette="bright", data=f1_results)
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xlabel("Type")
plt.xticks(rotation=45)
# plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.savefig("Results/f1_scores_plot.png")
plt.savefig("Results/f1_scores_plot_speculative.png")
plt.show()
exit()
for type in f1_results["type"].unique():
    sns.barplot(
        x="method",
        y="f1_score",
        hue="method",
        palette="Set1",
        data=f1_results[f1_results["type"] == type],
    )
    plt.title(f"F1 Score by Method for {type} Sentences")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xlabel("Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Results/f1_scores_{type}.png")
    plt.show()
