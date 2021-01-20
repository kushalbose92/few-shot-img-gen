## A Neighborhood-guided Few-Shot Generative Network

The project contains 6 .py files

* Files List
  * __dataloader.py__: Loads the datasets MNIST, fashion-MNIST and Omniglot
  * __model.py__: Proposed Model definition
  * __train.py__: Model training method
  * __eval.py__: Model evaluation method
  * __utils.py__: Utility methods
  * __main.py__: Driver functions 

* Input Parameters
  * num_shot = 5 
  * num_query = 5   
  * num_ways = 1 
  * alpha_val = 0.0
  * beta_val = 300.0
  * num_support = num_shot * num_ways
  * num_test = num_query * num_ways
  * num_samples = (num_shot + num_query) * num_ways
  * latent_dims = 2
  * num_train_tasks = 50000
  * num_test_tasks = 500
  * device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

