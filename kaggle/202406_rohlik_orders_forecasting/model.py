import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from tqdm import tqdm

def train_lightgbm(df: pd.DataFrame, target_column: str, date_column: str,
                        validation_size: float = 0.2, **kwargs) -> LGBMRegressor:
    """
    Train an XGBoost regression model.

    Parameters:
        df (pd.DataFrame):
        target_column (str): 
        validation_size (float): 
    Returns:
        model: The trained XGBoost model.
    """

    x = df.drop(columns=[target_column, date_column])
    y = df[target_column]

    if "trial" in kwargs and type(kwargs["trial"]) == optuna.Trial:
        # https://github.com/optuna/optuna/discussions/3930
        trial = kwargs["trial"]
        params = {
            "metric": "l1", # mae
            "n_estimators" : trial.suggest_int("n_estimators", 1, 2000),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            'max_depth':trial.suggest_int('max_depth', 2, 15),
            'num_leaves':trial.suggest_int('num_leaves', 2, 100),
            'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
            'learning_rate':trial.suggest_float('learning_rate',0.005,0.5, log=True),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.1, 1, step=0.01),
            "verbose":-1,
            "deterministic": True
        }

    else:
        # params = {
        #     'n_estimators': 1930,
        #     'boosting_type': 'gbdt',
        #     'max_depth': 4,
        #     'num_leaves': 59,
        #     'min_child_weight': 1,
        #     'learning_rate': 0.015261378053021128,
        #     'colsample_bytree': 0.88,
        #     'verbose':-1}
        # params = {"metric": "l1",
        #           'n_estimators': 940,
        #         'boosting_type': 'gbdt',
        #         'max_depth': 5,
        #         'num_leaves': 46,
        #         'min_child_weight': 2,
        #         'learning_rate': 0.41864240091047483,
        #         'colsample_bytree': 0.49,
        #         'verbose':-1}
        # params = {"metric": "l1",
        #           'n_estimators': 1925,
        #         'boosting_type': 'gbdt',
        #         'max_depth': 5,
        #         'num_leaves': 35,
        #         'min_child_weight': 3,
        #         'learning_rate': 0.017984814082637628,
        #         'colsample_bytree': 0.28,
        #         'verbose':-1}
        params = {"verbose":-1}

    model = LGBMRegressor(**params)
    model.fit(x, y)

    return model


def predict_with_model(model: LGBMRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        model (XGBRegressor): _description_
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    predictions = model.predict(df)
    result_df = df.copy()
    result_df['predictions'] = predictions
    return result_df


def add_features(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    train_data = train_data.copy(deep=True)
    test_data = test_data.copy(deep=True)
    for data in [train_data, test_data]:
        data["month_sin"] = data["timestamp"].dt.month.apply(lambda x: np.sin(x))
        data["month_cos"] = data["timestamp"].dt.month.apply(lambda x: np.cos(x))
        data["day_sin"] = data["timestamp"].dt.day.apply(lambda x: np.sin(x))
        data["day_cos"] = data["timestamp"].dt.day.apply(lambda x: np.cos(x))
        # data["hour_sin"] = data["date"].dt.hour.apply(lambda x: np.sin(x))
        # data["hour_cos"] = data["date"].dt.hour.apply(lambda x: np.cos(x))
    return train_data, test_data


class FeatureEngine:
    def __init__(self, energy: str):
        self.energy = energy
    
    def __call__(self, *args):
        return add_features(*args)



def prediction_process(data_train: pd.DataFrame, data_test: pd.DataFrame,
                       date_cut: str=None, prediction_horizon: int=48, timestep: int=1,
                       freq: str="H", energy = "Photovoltaic", **kwargs):
    """Iterative prediction process

    Args:
        data_train (pd.DataFrame): Dataframe with timestamp and target
        data_test (pd.DataFrame): Dataframe with timestamp and target
        date_cut (str, optional): First day of data for first prediction. Defaults to None.
        prediction_horizon (int, optional): _description_. Defaults to 48.
        timestep (int, optional): _description_. Defaults to 1.
        freq (str, optional): Frequency of the data. Unused for now. Defaults to "H".
        energy (str, optional): Energy type. Defaults to "Photovoltaic".

    Raises:
        ValueError: _description_
    """
    data_train = data_train[["timestamp", "target"]]
    data_test = data_test[["timestamp", "target"]]
    date_cut = max(data_train["timestamp"]) if date_cut is None else pd.to_datetime(date_cut)
    date_end = max(data_test["timestamp"]) if not data_test.empty else max(data_train["timestamp"])

    data_complete = pd.concat([data_train, data_test]) if not data_test.empty else data_train.copy(deep=True)
    data_complete = data_complete.sort_values(by=["timestamp"])

    feature_engine = FeatureEngine(energy)

    complete_predictions = []
    # for date_cut in tqdm(pd.date_range(start=date_cut, end=date_end - pd.DateOffset(hours=1),
    #                                    freq=freq.lower())):
    while date_cut < date_end:
        # print(date_cut)
        # Splitting into training and test data
        data_train = data_complete[data_complete["timestamp"] <= date_cut].reset_index(drop=True)
        data_test = data_complete[data_complete["timestamp"] > date_cut].reset_index(drop=True)
        if data_train["target"].isnull().any():
            raise ValueError("NaN in train")

        # Setting the prediction horizon
        new_prediction_horizon = prediction_horizon if len(data_test) > prediction_horizon \
                                                    else len(data_test) # this is just for autogluon

        data_train_model = data_train.copy(deep=True)
        # data_test_model = data_test[data_test["timestamp"] <= date_cut + pd.DateOffset(hours=new_prediction_horizon)].copy(deep=True)
        # data_test_model = data_test[data_test["timestamp"] <= date_cut + pd.DateOffset(days=new_prediction_horizon)].copy(deep=True)
        data_test_model = data_test.iloc[:new_prediction_horizon, :].copy(deep=True)

        # Adding features (if static, we can just add them to the dataframe beforehand)
        data_train_model, data_test_model = feature_engine(data_train_model, data_test_model)

        # Training the model
        model = train_lightgbm(data_train_model, target_column="target",
                                    date_column='timestamp', **kwargs) # it could be any train function
        # model = train_model(data_train_model, new_prediction_horizon)

        # Predicting
        predictions = predict_with_model(model, data_test_model.drop(columns=["timestamp", "target"])) # it could be any predict function
        # predictions = predict_model(model, data_train_model, data_test_model)
        predictions["timestamp"] = data_test_model["timestamp"]
        predictions["date_cut"] = date_cut
        predictions = predictions[["timestamp", "date_cut", "predictions"]]
        complete_predictions.append(predictions)

        # Updating data for next iteration
        predictions = predictions.rename(columns={"predictions": "target"})[["timestamp", "target"]]
        null_target_test = data_test.dropna(subset=["target"])[["timestamp", "target"]]
        if not null_target_test.empty:
            predictions = predictions[predictions["timestamp"]>max(null_target_test["timestamp"])]
            predictions = pd.concat([null_target_test, predictions])

        data_test = predictions.set_index("timestamp").reindex(data_test["timestamp"]).reset_index()

        data_complete = pd.concat([data_train, data_test])
        data_complete = data_complete.sort_values(by=["timestamp"])

        # date_cut += pd.DateOffset(hours=timestep)
        date_cut += pd.DateOffset(days=timestep)

    complete_predictions = pd.concat(complete_predictions)
    return complete_predictions

working_dir = "/kaggle/input/rohlik-orders-forecasting-challenge"
working_dir = "data/"

train = pd.read_csv(f"{working_dir}/train.csv")
train["date"] = pd.to_datetime(train["date"])
train_calendar = pd.read_csv(f"{working_dir}/train_calendar.csv")
train_calendar["date"] = pd.to_datetime(train_calendar["date"])
test = pd.read_csv(f"{working_dir}/test.csv")
test["date"] = pd.to_datetime(test["date"])

# train = train[["date","orders"]].rename(columns={"orders":"target", "date":"timestamp"})
train = train.rename(columns={"orders":"target", "date":"timestamp"})
# test["orders"] = pd.NA
# test["orders"] = test["orders"].astype(float)
test = test.reindex(columns=test.columns.tolist() + ["orders"])
# test = test[["date", "orders"]].rename(columns={"orders":"target", "date":"timestamp"})
test = test.rename(columns={"orders":"target", "date":"timestamp"})

warehouse = "Budapest_1"
train = train[train["warehouse"]==warehouse]
test = test[test["warehouse"]==warehouse]

full_predictions = pd.DataFrame()
for warehouse in tqdm(train["warehouse"].unique()):
    predictions = prediction_process(train[train["warehouse"]==warehouse].copy(deep=True),
                                     test[test["warehouse"]==warehouse].copy(deep=True),
                                     prediction_horizon=61, timestep=61, freq="D")
    predictions["warehouse"] = warehouse
    full_predictions = pd.concat([full_predictions, predictions]) if not full_predictions.empty\
                                                                  else predictions
full_predictions