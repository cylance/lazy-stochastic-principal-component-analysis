echo "run linear model"
python single_model.py linear RP
python single_model.py linear PCA
python single_model.py linear SPCA
python single_model.py linear SSPCA

echo "run random forest model"
python single_model.py rf10 RP
python single_model.py rf10 PCA
python single_model.py rf10 SPCA
python single_model.py rf10 SSPCA

echo "run time"
python run_time.py RP
python run_time.py PCA
python run_time.py SPCA
python run_time.py SSPCA

echo "compute subspace distance"
python compute_subspace_distance.py
python compute_Steifel_distance.py
python compute_Grassman_distance.py

echo "plot"
python plot_model_perf.py
python plot_run_time.py
python plot_subspace_distance.py
python plot_Steifel_distance.py
python plot_Grassman_distance.py