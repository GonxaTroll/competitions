import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def delete_null_columns(data: pd.DataFrame, proportion: float=0.7, cols_to_exclude: List = []) -> pd.DataFrame:
    nulls = data.isnull().sum() / len(data)
    selected_cols = data.columns[nulls < proportion]
    null_cols =  data.columns[nulls >= proportion]
    cols_to_exclude = [x for x in cols_to_exclude if x not in selected_cols]
    null_cols = null_cols.difference(cols_to_exclude)
    print(f"{len(null_cols)} columns deleted: {null_cols}")
    return data.loc[:, selected_cols.tolist() + cols_to_exclude]


def plot_null_percent(data: pd.DataFrame, proportion: float=0.7):
    plt.figure(figsize=(35,5))
    nulls = (data.isnull().sum() / len(data)).sort_values(ascending=False)
    nulls = pd.DataFrame(nulls).reset_index()
    nulls.columns = ["variable", "null_percentage"]
    sns.barplot(data=nulls, x="variable", y="null_percentage")
    plt.xticks(rotation=45)
    # plt.hlines(y=[proportion], linestyles=["-"], xmin=0, xmax=1)
    # fig.refline(y=proportion)
    plt.show()

def get_column_descriptions(data: pd.DataFrame) -> pd.DataFrame:
    definitions_path = "data/home-credit-credit-risk-model-stability/feature_definitions.csv"
    feature_definitions = pd.read_csv(definitions_path)
    return feature_definitions[feature_definitions["Variable"].isin(data.columns.tolist())]


def order_columns_alphabetically(data: pd.DataFrame) -> pd.DataFrame:
    initial_columns = [x for x in ["case_id", "num_group1", "num_group2"]
                       if x in data.columns]
    other_columns = [x for x in data.columns.sort_values()
                     if x not in initial_columns]
    data = data[initial_columns + other_columns]
    return data


def read_sets_of_dataframes(path: str, names: List):
    data_list = []
    for name in names:
        data_list.append(pd.read_parquet(f"{path}{name}.parquet"))
    return data_list


def convert_date_columns(data: pd.DataFrame, names: List):
    for col in names:
        if str(data[col].dtype) in ["object", "categorical"]:
            try:
                data[col] = pd.to_datetime(data[col], format="%Y-%m-%d")
            except Exception:
                print(col)


def compute_chi2(data: pd.DataFrame, plot=False):
    from scipy.stats import chi2_contingency
    categorical_v = data.columns[(data.dtypes == "object") | (data.dtypes == "bool")]
    chi2_df = []
    for col in categorical_v:
        contingency = pd.crosstab(index=data["target"],
                                  columns = [data[col]])
        _, pvalue, _, _ = chi2_contingency(contingency.values)
        chi2_df.append([col, pvalue])

        if plot:
            counts = data[col].value_counts(normalize=True).reset_index()
            plt.figure(figsize=(25,5))
            plt.subplot(1,2,1)
            sns.barplot(data=counts, x=col, y="proportion")
            counts = data.groupby(["target"])[col].value_counts(normalize=True).reset_index()
            plt.subplot(1,2,2)
            sns.barplot(data=counts, x=col, y="proportion", hue="target")
            plt.suptitle(f"Chi-square {col}: p-value = {pvalue}")
            plt.show()

    chi2_df = pd.DataFrame(chi2_df, columns=["variable", "pvalue"])
    return chi2_df


def select_by_chi2(data: pd.DataFrame, cols_to_exclude: List=[]):
    chi2_df = compute_chi2(data)
    selected_variables = chi2_df[chi2_df["pvalue"]<0.05]["variable"].tolist()
    cols_to_exclude = [x for x in cols_to_exclude if x not in selected_variables]
    print(f"Not selected variables: {chi2_df[chi2_df['pvalue']>=0.05]['variable'].tolist()}")
    selected_variables += cols_to_exclude
    other_variables = data.columns.difference(chi2_df["variable"].tolist()).tolist()
    return data[other_variables+selected_variables]


### UTILS ###
def reduce_column_size(data: pd.DataFrame):
    for col in data.columns:
        dtype = str(data[col].dtype)
        if "int" in dtype:
            max_abs_value = abs(data[col]).max()
            for exp in [8, 16, 32, 64]:
                if 2**(exp-1) >= max_abs_value:
                    data[col] = data[col].astype(f"int{exp}")
                    break
        elif "float" in dtype:
            max_abs_value = abs(data[col]).max()
            for exp in [32, 64]:
                if 2**(exp-1) >= max_abs_value:
                    data[col] = data[col].astype(f"float{exp}")
                    break
    return data


def categorical_to_dummies(data: pd.DataFrame, dtype=8):
    if dtype is None:
        dtype = 64
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == "object":
            if data[col].isnull().sum() == 0:
                drop_first = True
            else:
                drop_first = False
            data = pd.concat([data, pd.get_dummies(data[col], dtype=f"int{dtype}",
                                                   drop_first=drop_first, prefix=f"{col}_")],
                            axis=1)
            categorical_cols.append(col)
    data = data.drop(columns=categorical_cols)
    return data

INDEX_COLUMNS = ["case_id", "num_group1", "num_group2", "date_decision", "MONTH", "WEEK_NUM",
                 "target", "score"]

from sklearn.feature_selection import VarianceThreshold
def delete_constant_columns(data: pd.DataFrame, **kwargs):
    # Numerical features
    variance_threshold = 0
    numerical_data = get_dataset_by_datatype(data, "numerical", exclude_columns=INDEX_COLUMNS)
    numerical_cols_delete = []
    if not numerical_data.empty:
        if "variance_threshold" in kwargs:
            variance_threshold = kwargs["variance_threshold"]
        var_threshold = VarianceThreshold(threshold=variance_threshold)
        var_threshold.fit(numerical_data)
        numerical_variable_columns = var_threshold.get_feature_names_out().tolist()
        numerical_cols_delete = [x for x in numerical_data.columns
                                if x not in numerical_variable_columns]

    # Categorical and datetime
    minimum_number_categories = 1
    categorical_columns = get_columns_by_datatype(data, "categorical",
                                                  exclude_columns=INDEX_COLUMNS)
    date_columns = get_columns_by_datatype(data, "date", exclude_columns=INDEX_COLUMNS)
    categorical_columns = categorical_columns.union(date_columns)
    categorical_cols_delete = []
    if len(categorical_columns):
        if "minimum_number_categories" in kwargs:
            minimum_number_categories = kwargs["minimum_number_categories"]
        for col in categorical_columns:
            if len(data[col].unique()) <= minimum_number_categories:
                categorical_cols_delete.append(col)
    cols_delete = numerical_cols_delete + categorical_cols_delete
    print(f"Number of constant columns deleted: {len(cols_delete)}: {cols_delete}")
    return data.drop(columns=cols_delete)


import numpy as np
def get_columns_by_datatype(data: pd.DataFrame, column_type="categorical",
                            extra_columns = list(), exclude_columns = list()):
    extra_columns = data.columns.intersection(extra_columns)
    exclude_columns = data.columns.intersection(exclude_columns)
    if column_type == "categorical":
        column_type = [np.dtype("O")]
    elif column_type == "numerical":
        column_type = [np.dtype(f"{number_type}{precision}") for number_type in ["int", "float"]
                                                             for precision in [32, 64]]
        column_type += ["int8", "int16"]
    elif column_type == "date":
        column_type = [np.dtype('datetime64[ns]'), np.dtype('<M8[ns]')]
    columns = data.columns[np.isin(data.dtypes.values, column_type)]
    columns = columns.difference(exclude_columns)
    columns = extra_columns.union(columns)
    return columns


def get_dataset_by_datatype(data: pd.DataFrame, column_type="categorical",
                        extra_columns = list(), exclude_columns = list()):
    columns = get_columns_by_datatype(data, column_type, extra_columns, exclude_columns)
    return data[columns].copy(deep=True)


def transform_location_info(data: pd.DataFrame):
    def aux_transform_zip_code(zip: str):
        try:
            return int(zip[1:])
        except:
            return pd.NA
    zip_code = data["contaddr_zipcode_807M"].str.split("_", expand=True)
    zip_code.loc[:, 0] = zip_code.loc[:, 0].map(aux_transform_zip_code)
    zip_code_cols = []
    for col in range(len(zip_code.columns)):
        zip_code[col] = pd.to_numeric(zip_code[col], errors="coerce").astype("float64")
        zip_code_cols.append(f"zip_code_part_{col+1}")
    zip_code.columns = zip_code_cols

    data = data.drop(columns=[x for x in data.columns
                              if "cont" in x or "zipcode" in x or "registaddr" in x])
    data = pd.concat([data, zip_code], axis=1)
    return data


def agg_categories(data: pd.DataFrame, column: str, n_top_categories: int=4,
                   train_dataset: pd.DataFrame = None):
    if train_dataset is not None:
        top_categories = train_dataset[column][train_dataset[column]!="OTHER"]
    else:
        top_categories = data[column].value_counts()[:n_top_categories].index.tolist()
    def agg_categories_aux(value, top_categories):
        if value is not None and value not in top_categories: ############################3 how to check for null values
            return "OTHER"
        return value
    return data[column].map(lambda x: agg_categories_aux(x, top_categories))


def mix_cols_parallel(data: pd.DataFrame, original_cols: list):
    list_cols = []
    new_values = []
    for col in original_cols:
        list_cols.append(data[col].tolist())
    for i in range(len(list_cols[0])):
        row_cols = pd.Series([list_cols[x][i] for x in range(len(list_cols))])
        row_cols = row_cols.dropna()
        if not len(row_cols):
            new_values.append(pd.NA)
        else:
            unique_values = row_cols.unique()
            if len(unique_values) > 1:
                value_counts = row_cols.value_counts(normalize=True)
                majority = value_counts[value_counts>0.5]
                if not majority.empty:
                    new_values.append(majority.iloc[0])
                else:
                    if type(unique_values[0]) == str:
                        new_values.append(unique_values[0])
                    else:
                        try:
                            row_cols = pd.to_datetime(row_cols)
                        except:
                            pass
                        try:
                            new_values.append(row_cols.mean())
                        except:
                            new_values.append(unique_values[0])
                            print(i)
            else:
                new_values.append(unique_values[0])
    return new_values


def get_highly_correlated_features(data: pd.DataFrame, threshold: float=0.9):
    # get the numerical data here or before
    linear_corrs = data.corr()
    linear_corrs = linear_corrs.unstack().reset_index()
    linear_corrs.columns = ["v1", "v2", "corr"]

    def reorder_corr_values(row: pd.Series):
        if row["v1"] > row["v2"]:
            row["v1"], row["v2"] = row["v2"], row["v1"]
        return row

    linear_corrs = linear_corrs.apply(reorder_corr_values, axis=1)
    linear_corrs = linear_corrs.drop_duplicates()
    linear_corrs = linear_corrs[linear_corrs["v1"] != linear_corrs["v2"]]

    def get_acc_correlation(linear_corrs: pd.DataFrame):
        linear_corrs = pd.concat([linear_corrs[["v1","corr"]].rename(columns={"v1":"feature"}),
                                linear_corrs[["v2","corr"]].rename(columns={"v2":"feature"})
                                ])
        linear_corrs = linear_corrs.groupby(["feature"]).sum().reset_index()
        return linear_corrs
    
    features_to_delete = []
    while not linear_corrs[linear_corrs["corr"]>=threshold].empty:
        highly_correlated = linear_corrs[linear_corrs["corr"]>=threshold]
        highly_correlated = highly_correlated.melt(value_vars=["v1", "v2"])
        highly_correlated = highly_correlated["value"].value_counts()\
                                                      .reset_index()\
                                                      .rename(columns={"index":"feature"})

        # Comparing the repeated features and getting the ones with most correlation sum
        accumulated_corrs = get_acc_correlation(linear_corrs)
        highly_correlated = highly_correlated.merge(accumulated_corrs, on=["feature"])
        highly_correlated = highly_correlated.sort_values(by=["value", "corr"],
                                                          ascending=[False, False])

        selected_feature = highly_correlated.iloc[0]["feature"]
        features_to_delete.append(selected_feature)

        # Removing the feature from the correlation matrix
        linear_corrs = linear_corrs[((linear_corrs["v1"]!=selected_feature) &
                                     (linear_corrs["v2"]!=selected_feature))]
    return features_to_delete


def remove_highly_correlated_features(data: pd.DataFrame, threshold: float = 0.9):
    highly_correlated =\
        get_highly_correlated_features(data=get_dataset_by_datatype(data,
                                                                    "numerical",
                                                                    exclude_columns=INDEX_COLUMNS),
                                       threshold=threshold)
    data = data.drop(columns=highly_correlated)
    return data


def select_categorical_with_many_categories(data: pd.DataFrame, cut_category_number: int=50,
                                            original_from_dummies: list = list()):
    cut_category_cols = []
    if len(original_from_dummies):
        for column in original_from_dummies:
            dummy_cols = data.columns[data.columns.str.startswith(column)]
            if len(dummy_cols) > cut_category_number:
                cut_category_cols += dummy_cols.tolist()
    else:
        for column in get_dataset_by_datatype(data, "categorical").columns:
            if len(data[column].value_counts()) > cut_category_number:
                cut_category_cols.append(column)
        print(f"Columns with over {cut_category_number} categories: {cut_category_cols}")
    return cut_category_cols
