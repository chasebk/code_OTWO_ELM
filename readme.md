# A Novel Workload Prediction Model Using ELM with Opposition-based Tug of War Optimization
[![GitHub release](https://img.shields.io/badge/release-2.0.0-yellow.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3626114.svg)](https://doi.org/10.5281/zenodo.3626114)
[![License](https://img.shields.io/packagist/l/doctrine/orm.svg)]()

## Dear friends and followers
* I updated to repository to the newest version, which is very easy to read and reproduce. 
* All of our optimizer (meta-heuristics) now deleted and taken the new one from my newest library: 

        https://pypi.org/project/mealpy/
    
* If you use my code or library in your project, I would appreciate the cites:
    * Nguyen, T., Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019). Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, 12(2), 1144-1161.
	
	* Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
	
    * Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

## How to read my repository
1. data: include raw and formatted data
2. envs: include conda environment and how to install conda environment 
3. utils: Helped functions such as IO, Draw, Math, Settings (for all model and parameters), Preprocessing...
4. model: (2 folders) 
    * root: (want to understand the code, read this classes first)
        * root_base.py: root for all models (traditional, hybrid and variants...) 
        * traditional: root for all traditional models (inherit: root_base)
        * hybrid: root for all hybrid models (inherit: root_base)
    * main: (final models)
        * this classes will use those optimizer above and those root (traditional, hybrid) above 
        * the running files (outside with the downloaded folder) will call this classes
        * the traditional models will use single file such as: traditional_elm, traditional_rnn,...
        * the hybrid models will use 2 files, example: hybrid_elm.py and GA.py (optimizer files)
6. special files
    * *_scipt.py: running files (*: represent model) such as ga_elm_script.py => GA + ELM
    
    
## Notes

1. To improve the speed of Pycharm when opening (because Pycharm will indexing when opening), you should right click to 
paper and data folder => Mark Directory As  => Excluded

2. How to run models? 
```code 
1. Before runs the models, make sure you clone this repository in your laptop:
    https://github.com/chasebk/code_OTWO_ELM

2. Then open it in your editor like Pycharm or Spider...

3. Now you need to create python environment using conda (assumpted that you have already had it). Open terminal
    conda  your_environment_name  create -f  envs/env.yml           (go to the root project folder and create new environment from my file in: envs/env.yml)

4. Now you can activate your environment and run the models
    conda activate your_environment_name        # First, activate your environment to get the needed libraries.
    python model_name

For example:
    conda ai_env create -f envs/env.yml
    conda activate ai_env
    python elm_script.py

5. My model name:

    1. MLNN (1 HL) => mlnn1hl_script.py
	2. ELM => elm_script.py
	3. GA-ELM => ga_elm_script.py
	4. PSO-ELM => pso_elm_script.py
	5. ABFOLS-ELM => abfols_elm_script.py
	6. DE-ELM => de_elm_script.py
	7. TWO-ELM => two_elm_script.py
	8. OTWO-ELM => otwo_elm_script.py
```

### How to change model's parameters?
```code 
You can change the model's parameters in file: utils/SettingPaper.py 

For example:

+) For traditional models: MLNN, ELM

####: MLNN-1HL
mlnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [5000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

- If you want to tune the parameters, you can adding more value in each parameters like this:

- sliding: [1, 2, 3, 4] or you want just 1 parameter: [12]
- hidden_sizes: [ [5], [10], [100] ] or [ [14] ]
- activations: [ ("elu", "relu"), ("elu", "tanh") ] or just: [ ("elu", "elu") ]
- learning_rate: [0.1, 0.01, 0.001] or just: [0.1]
....


+ For hybrid models: GA_ELM, PSO_ELM, ABFOLS-ELM, TWO-ELM, OTWO-ELM
 
#### : GA-ELM
ga_elm_paras_final = {
    "sliding": [2, 3, 5],
    "hidden_size" : [(5, False)],
    "activation": ["elu"],                  
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [100],
    "pop_size": [20],                   # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}

- Same as traditional models.
```

### Where is the results folder?
```code 
- Look at the running file, for example: ga_elm_script.py

+) For 1-time runs (Only run 1 time for each model).
    _There are 3 type of results file include: model_name.csv file, model_name.png file and Error-model_name.csv file 
    _model_name.csv included: y_true and y_predict 
    _model_name.png is visualized of: y_true and y_predict (test dataset)
    _Error-model_name.csv included errors of training dataset after epoch. 1st, 2nd column are: MSE errors

=> All 3 type of files above is automatically generated in folder which you can set in SettingPaper

+) For stability runs (Run each model n-time with same parameters).
Because in this test, we don't need to visualize the y_true and y_predict and also don't need to save y_true and y_predict
So I just save n-time running in the same csv files in folder

- Noted:

+ In the training set we use MSE. But for the testing set, we can use so much more error like: R2, MAPE, ...
You can find the function code described it in file: 
    model/root/root_base.py 
        + _save_results__: 
```

* Take a look at project_structure.md file.  Describe how the project was built.
