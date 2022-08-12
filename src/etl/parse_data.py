# parses all input data and stores the procesed data
import pandas as pd
import config
import os


def load_csv(input_path: str, *args, **kwargs) -> pd.DataFrame:

    dataset = pd.read_csv(input_path, *args, **kwargs)

    return dataset


def join_and_filter_items(sales: pd.DataFrame, products: pd.DataFrame, oil: pd.DataFrame,
                          holidays: pd.DataFrame) -> pd.DataFrame:


    # sales = sales[sales['store_nbr'] == 12]
    sales = sales.merge(products, how='inner', on='item_nbr')
    sales = sales.merge(oil, how='left', on='date')
    sales = sales.merge(holidays, how='left', on='date')
    sales = sales.drop(columns=['family', 'description'])
    return sales



def etl():
    # loads all csvs and join into one big master table
    # sales = load_csv(os.path.join(config.INPUT_FOLDER, 'train.csv'), usecols=[1, 2, 3, 4, 5])
    # sales = sales.groupby(['date', 'item_nbr']).agg(unit_sales=('unit_sales', 'sum'),
    #                                                 onpromotion=('onpromotion', 'mean')).reset_index()
    # sales.to_csv(os.path.join(config.PROCESSED_FOLDER, 'sales_aggr.csv'), index=False)
    sales = load_csv(os.path.join(config.PROCESSED_FOLDER, 'sales_aggr.csv'))
    items = load_csv(os.path.join(config.INPUT_FOLDER, 'items.csv'))
    items = items[items['family'] == 'CLEANING']
    oil = load_csv(os.path.join(config.INPUT_FOLDER, 'oil.csv'))
    holidays = load_csv(os.path.join(config.INPUT_FOLDER, 'holidays_events.csv'))
    # we ignore multiple holidays on 1 day, make days unique
    holidays = holidays.groupby('date').first()

    sales_filtered = join_and_filter_items(sales, items, oil, holidays)
    # sales_filtered.dropna(axis=0, inplace=True)
    sales_filtered.to_csv(os.path.join(config.PROCESSED_FOLDER, 'processed.csv'), index=False)
    print('Etl run successfully.')

if __name__ == '__main__':
    etl()
