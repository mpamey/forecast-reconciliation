import os


DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

INPUT_FOLDER = os.path.join(DATA_FOLDER, '0_input')
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, '1_processed')
MODEL_FOLDER = os.path.join(DATA_FOLDER, '2_model_output')
RECONCILE_FOLDER = os.path.join(DATA_FOLDER, '3_reconcile_output')

for fldr in [DATA_FOLDER, INPUT_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, RECONCILE_FOLDER]:
    if not os.path.exists(fldr):
        os.mkdir(fldr)