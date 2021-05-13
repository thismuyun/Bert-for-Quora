# Bert-for-Quora
Bert as service

# Bert Install 
    pip install tensorflow==1.15.0 
    pip install bert-serving-server  # server
    pip install bert-serving-client  # client, independent of `bert-serving-server`

# Bert service start

    bert-serving-start -model_dir model/wwm_uncased_L-24_H-1024_A-16 -num_worker=1
    
    
- num_worker CPU/GPU 工作的数量

    

