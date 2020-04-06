## Install Environments

1. Do it on your terminal (ubuntu) or bash shell (windows). Make sure you already have miniconda
```code 
Check this link for pre-installed package. Due to new version of tensorflow. Both windows and linux
    https://www.tensorflow.org/install/pip?lang=python3

After that:

First way: (Do it by yourself)
    conda --version     
    conda update conda 
    conda info --envs
    
    conda create -n ai python==3.7.6
    conda activate ai

    conda install pandas matplotlib scikit-learn scrapy seaborn
    conda install -c anaconda tensorflow
    conda install -c anaconda ipython-notebook
    conda install -c conda-forge statsmodels
    conda install keras  
    pip install mealpy
    

Second way: (create env by my env.yml file)

    conda ai create -f env.yml (the first line in that file is the name of the environment)
    conda activate ai
    pip install mealpy

```

2. Useful command
```code 

1) Activate, check and deactivate environment
    conda activate ai       

    conda list          (or)
    conda env list 
    conda info --envs    

    source deactivate           
    
2) Check package inside the environment 
    conda list -n ai                (if ai hasn't activated)
    conda list                      (if ai already activated)
    
3) Export to .yml for other usage.  
    source activate ai                  (access to environment)
    conda env export > env.yml     

4) Delete environment 
    conda remove --name ai --all     (or)
    conda env remove --name ai   
    
    conda info --envs   (kiểm tra xem đã xóa chưa)
```
