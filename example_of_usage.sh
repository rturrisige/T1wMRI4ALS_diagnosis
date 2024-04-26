X_DIR=path/to/input
Y_DIR=path/to/labels
SAVER_DIR=path/to/saver_folder
GS_CV=10

for DATA_TYPE in radiomics L7 L8
do
  python diagnosis_grid_search \
  --data_X $X_DIR \
  --data_Y $Y_DIR \
  --saver_dir SAVER_DIR \
  --input_type $DATA_TYPE \
  --n_splits GS_CV

  python diagnosis \
  --data_X $X_DIR \
  --data_Y $Y_DIR \
  --saver_dir SAVER_DIR \
  --input_type $DATA_TYPE \
  --n_splits 3 \
  --best_parameters "${SAVER_DIR}/${DATA_TYPE}_best_parameters_${GS_CV}CV.csv"
done
