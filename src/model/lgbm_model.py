import pandas as pd
import config
import os
import lightgbm as lgb
import pickle
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hts

def load_csv(input_path: str, *args, **kwargs) -> pd.DataFrame:

    dataset = pd.read_csv(input_path, *args, **kwargs)

    return dataset


def ordinal_label_encoder(df: pd.DataFrame) -> pd.DataFrame:
    oe_family = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df['type'] = oe_family.fit_transform(df['type'].values.reshape(-1, 1))
    df['locale'] = oe_family.fit_transform(df['locale'].values.reshape(-1, 1))
    df['locale_name'] = oe_family.fit_transform(df['locale_name'].values.reshape(-1, 1))
    df['transferred'] = oe_family.fit_transform(df['transferred'].values.reshape(-1, 1))

    return df


def train_test_split(train_data):

    # train = train_data[train_data['date'] <= '2017-05-31']#.set_index(['date', 'level'], drop=False)
    #
    # val = train_data[(train_data['date'] > '2017-05-31') & (train_data['date'] <= '2017-07-31')]#.set_index(['date', 'level'], drop=False)
    #
    # test = train_data[train_data['date'] > '2017-07-31']#.set_index(['date', 'level'], drop=False)

    train = train_data[train_data['date'] < '2016-01-01']  # .set_index(['date', 'level'], drop=False)

    val = train_data[(train_data['date'] >= '2016-01-01') & (
                train_data['date'] < '2017-01-01')]  # .set_index(['date', 'level'], drop=False)

    test = train_data[train_data['date'] >= '2017-01-01']  # .set_index(['date', 'level'], drop=False)

    return train, val, test



def train_and_predict_models(train_df, val_df, test_df):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    test_predicted = []

    for level in train_df['level'].unique():
        print(f'Working on lvl {level}')
        # filter data on level
        train_lvl_df = train_df[train_df['level'] == level]
        val_lvl_df = val_df[val_df['level'] == level]
        test_lvl_df = test_df[test_df['level'] == level]

        if len(test_lvl_df) > 0:
            # split and set index
            train_X, train_y = split_x_y(train_lvl_df)

            test_X, test_y = split_x_y(test_lvl_df)

            # prepare and train
            lgtrain = lgb.Dataset(train_X, label=train_y)

            if len(val_lvl_df) > 0:
                val_X, val_y = split_x_y(val_lvl_df)
                lgval = lgb.Dataset(val_X, label=val_y)
                evals_result = {}
                model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20,
                                  evals_result=evals_result)
            else:
                # if no data in validation set, no early stopping, just do 1000
                model = lgb.train(params, lgtrain, 1000)

            # predict
            test_y = predict_model(model, test_X, test_y)
            test_predicted.append(test_y)

    return pd.concat(test_predicted)


def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_indexed = df.set_index(['date', 'level'], drop=True)
    df_X = df_indexed.loc[:, ~df_indexed.columns.isin(['unit_sales'])]
    df_y = df_indexed.loc[:, df_indexed.columns == 'unit_sales']

    return df_X, df_y


def predict_model(model, test_X, test_y):
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    test_y['predictions'] = pred_test_y
    return test_y


def create_hierarchies(train_data):
    # we create different hierarchies
    item_df = train_data.copy()

     # class level (middle level)
    class_df = train_data.groupby(['date', 'class']).agg(unit_sales=('unit_sales', 'sum'),
                                                         onpromotion=('onpromotion', 'mean'),
                                                         perishable=('perishable', 'mean'),
                                                         # all same for date:
                                                         dcoilwtico=('dcoilwtico', lambda x: x.iloc[0]),
                                                         type=('type', lambda x: x.iloc[0]),
                                                         locale=('locale', lambda x: x.iloc[0]),
                                                         locale_name=('locale_name', lambda x: x.iloc[0]),
                                                         transferred=('transferred', lambda x: x.iloc[0])
                                                         ).reset_index()

    # top lvl (top level)
    total_df = train_data.groupby('date').agg(unit_sales=('unit_sales', 'sum'),
                                              onpromotion=('onpromotion', 'mean'),
                                              perishable=('perishable', 'mean'),
                                              # all same for date:
                                              dcoilwtico=('dcoilwtico', lambda x: x.iloc[0]),
                                              type=('type', lambda x: x.iloc[0]),
                                              locale=('locale', lambda x: x.iloc[0]),
                                              locale_name=('locale_name', lambda x: x.iloc[0]),
                                              transferred=('transferred', lambda x: x.iloc[0])
                                              ).reset_index()

    # add level identifiers
    item_df['level'] = item_df.apply(lambda x: f"class_{x['class']}-item_{x['item_nbr']}", axis=1)
    # within a level the item is always the same so we drop item_nbr
    item_df = item_df.drop(columns=['item_nbr'])
    class_df['level'] = class_df['class'].apply(lambda x: f'class_{x}')
    # same for class in class aggr
    class_df = class_df.drop(columns=['class'])
    total_df['level'] = 'total'

    df_hierarchy = pd.concat([item_df, class_df, total_df])

    return df_hierarchy


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:

    # df['B_shifted'] = df.sort_values(by=['level', 'date']).groupby(['level'])['unit_sales'].transform('shift', 2)
    df_ext = df.sort_values(by=['level', 'date'])
    df_ext['unit_sales_shift1'] = df_ext.groupby(['level'])['unit_sales'].shift(1)
    df_ext['unit_sales_shift2'] = df_ext.groupby(['level'])['unit_sales'].shift(2)
    df_ext['unit_sales_shift7'] = df_ext.groupby(['level'])['unit_sales'].shift(7)

    return df_ext


def cast_categorical_features(df):

    obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)

    for feature in obj_feat:
        df[feature] = pd.Series(df[feature], dtype="category")

    return df


def level_to_level_type(level):
    if level == 'total':
        return 'total'
    elif 'item' in level:
        return 'item'
    else:
        return 'class'


def bottumup_approach(base_forecast_df, labels):
    reconciled_df = base_forecast_df.copy()
    all_columns = base_forecast_df.columns
    for col in all_columns:
        if 'item' in col:
            # in bottomup we dont touch the lowest level
            continue
        elif 'class' in col:  # and not item ofcourse
            # for each class, find all item children and sum those
            class_children_cols = [c for c in all_columns if col in c and 'item' in c]
            reconciled_df[col] = base_forecast_df.loc[:, class_children_cols].sum(axis=1)
        elif 'total' in col:
            # for the total, find all item cols to sum
            total_children_cols = [c for c in all_columns if 'item' in c]
            reconciled_df[col] = base_forecast_df.loc[:, total_children_cols].sum(axis=1)
        else:
            assert False, 'error we dont except'

    reconciled_df = reconciled_df[labels]

    return reconciled_df.to_numpy()  # to be consistent with other methods in terms of output


def modeling(train=True):
    if train:
        train_data = load_csv(os.path.join(config.PROCESSED_FOLDER, 'processed.csv'), parse_dates=['date'])

        # train_data = ordinal_label_encoder(train_data)
        train_hierarchical = create_hierarchies(train_data)

        # set categorical variables
        train_converted = cast_categorical_features(train_hierarchical)

        train, val, test = train_test_split(train_converted)

        train_ext = add_lagged_features(train)
        val_ext = add_lagged_features(val)
        test_ext = add_lagged_features(test)

        test_predicted = train_and_predict_models(train_ext, val_ext, test_ext)

        print('Trained model successfully.')
        # test_y = predict_model(model, test_X, test_y)
        # test_X['predictions'] = test_y
        # print('Predicted testdata successfully.')
        test_predicted.to_csv(os.path.join(config.MODEL_FOLDER, 'forecasts.csv'), index=True)
        print('Saved predictions and successfully.')

    test_predicted = load_csv(os.path.join(config.MODEL_FOLDER, 'forecasts.csv'))
    test_predicted['residuals'] = test_predicted['unit_sales'] - test_predicted['predictions']

    # now we look into reconciliationg

    unique_levels = list(test_predicted['level'].unique())
    unique_classes = [l for l in unique_levels if 'class' in l and '-' not in l]
    unique_class_items = [l for l in unique_levels if 'item' in l]

    total = {'total': unique_classes}
    classes = {k: [v for v in unique_class_items if v.startswith(k)] for k in unique_classes}
    hierarchy = {**total, **classes}

    predicted_pivoted = test_predicted.pivot(index="date", columns="level", values="predictions")
    actuals_pivoted = test_predicted.pivot(index="date", columns="level", values="unit_sales")
    residuals_pivoted = test_predicted.pivot(index="date", columns="level", values="residuals")
    squared_residuals = residuals_pivoted**2
    mse = squared_residuals.mean(axis=0).to_dict()

    # nan values are causing issues
    # for now we assume that nans in the test set are from products that were not sold
    # we therefore set actual to 0 and also prediction to 0 for these product date combinations
    nan_indexes = actuals_pivoted.isnull()
    actuals_pivoted[nan_indexes] = 0
    predicted_pivoted[nan_indexes] = 0

    # # build the hierarchy tree as a dictionary
    tree = hts.hierarchy.HierarchyTree.from_nodes(nodes=hierarchy, df=actuals_pivoted)
    sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)

    pred_dict = {}
    ordered_mse = {}
    for label in sum_mat_labels:
        # place the predictions in the right order in a dict. key: level, value: list[values]
        pred_dict[label] = pd.DataFrame(data=predicted_pivoted[label].values, columns=['yhat'])
        # also order the mse in the order as specified in sum_mat_labels
        # note that any dict is ordered from python 3.7 onwards
        ordered_mse[label] = mse[label]

    total_results = None
    for method in ['BU', 'OLS', 'WLSV']:

        if method == 'BU':
            # do something
            reconciliated_predictions = bottumup_approach(predicted_pivoted, sum_mat_labels)
        else:
            # note that ordered_mse is not used when OLS is the method, it just gets ignored
            reconciliated_predictions = hts.functions.optimal_combination(pred_dict, sum_mat, method=method, mse=ordered_mse)


        # revised into long format again
        recon_annotated = pd.DataFrame(reconciliated_predictions, columns=sum_mat_labels, index=predicted_pivoted.index).reset_index()
        recon_annotated = recon_annotated.melt(id_vars=['date'], var_name='level')
        recon_annotated = recon_annotated.rename(columns={"value": "predicted_reconcil"})

        final_predictions = test_predicted.merge(recon_annotated, on=['date', 'level'], how='inner')

        final_predictions['level_type'] = final_predictions['level'].apply(lambda x: level_to_level_type(x))
        # now we evaluate:
        if total_results is None:
            normal_rmse = final_predictions.groupby('level_type').apply(lambda x: mean_squared_error(x['unit_sales'], x['predictions'])**.5)
            total_results = pd.DataFrame([normal_rmse], index=['normal_rmse'])

        recon_rmse = final_predictions.groupby('level_type').apply(
            lambda x: mean_squared_error(x['unit_sales'], x['predicted_reconcil']) ** .5)

        recon_rmse = pd.DataFrame([recon_rmse], index=[f'{method}_rmse'])
        total_results = pd.concat([total_results, recon_rmse])
        # total_results = pd.DataFrame([total_results, recon_rmse], index=['normal_rmse', 'recon_rmse'])
    print(total_results.loc[:, ['item', 'class', 'total']])
    print('goodbye')


if __name__ == '__main__':
    modeling(train=False)
