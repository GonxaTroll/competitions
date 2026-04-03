import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def compute_chi2(data: pd.DataFrame, categorical_v = None) -> pd.DataFrame:
    """Computes the chi-square test for categorical variables.

    Args:
        data (pd.DataFrame): Dataframe with categorical variables.
        categorical_v (list, optional): List of categorical variables to include in the test.
                                        If None, all string-type columns are used.

    Returns:
        pd.DataFrame: Dataframe with chi-square test results.
    """

    # categorical_v = data.columns[(data.dtypes == "object") | (data.dtypes == "bool")]
    if categorical_v is None:
        categorical_v = [
        x for x in data.columns if (pd.api.types.is_string_dtype(data[x]))
        ]
    chi2_df = []
    for col in categorical_v:
        contingency = pd.crosstab(index=data["target"],
                                  columns = [data[col]])
        _, pvalue, _, _ = chi2_contingency(contingency.values)
        chi2_df.append([col, pvalue])

    chi2_df = pd.DataFrame(chi2_df, columns=["variable", "pvalue"])
    return chi2_df



def plot_null_percent(data: pd.DataFrame, proportion: float = 0.7):
    """Plots the percentage of null values in each column of a dataframe.

    Args:
        data (pd.DataFrame): Dataframe from which to compute the percentage of null values.
        proportion (float, optional): Proportion of null values to show as threshold.
                                      Defaults to 0.7.
    """
    plt.figure(figsize=(35,5))
    nulls = (data.isnull().sum() / len(data)).sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    nulls = pd.DataFrame(nulls).reset_index()
    nulls.columns = ["variable", "null_percentage"]
    ax = sns.barplot(data=nulls, x="variable", y="null_percentage")
    plt.xticks(rotation=45)

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.001,
                f'{height:.3%}',
                ha="center")
    if max(nulls["null_percentage"]) >= proportion:
        plt.hlines(y = proportion, linestyles="dashed", xmin = -0.5, xmax = len(nulls)-0.5,
                   colors = "red")
    # return fig
    plt.show()
