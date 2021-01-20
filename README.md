## A Simple Neighborhood-guided Few-Shot Generative Network

The project contains 6 .py files

* Files list:
  * __dataloader.py__: Loads the datasets MNIST, fashion-MNIST and Omniglot
  * __model.py__: Proposed Model definition
  * __train.py__: Model training method
  * __eval.py__: Model evaluation method
  * __utils.py__: Utility methods
  * __main.py__: Driver functions 

* Input Parameters:
  * __num_shot__ = 5  (Number of training samples from every class in each task)
  * __num_query__ = 5 (Number of query samples from every class in each task)  
  * __num_ways__ = 3  (Number of classes in each task)
  * __alpha_val__ = 50.0 (Weight for intra-cluster loss)
  * __beta_val__ = 20.0 (Weight for inter-clustering loss) 
  * __num_support__ = num_shot * num_ways (Total training samples in atask)
  * __num_test__ = num_query * num_ways (Total query samples in a task)
  * __num_samples__ = (num_shot + num_query) * num_ways (Totalnumber of samples in a single task)
  * __latent_dims__ = 2 (Dimension of latent space)
  * __num_train_tasks__ = 50000 (Number of training tasks)
  * __num_test_tasks__ = 500 (Number of test tasks)
  * __dataset__ = "fashion-mnist (Name of the dataset)
  * __use_saved_model__ = False (Whether to use saved model or not)
  * __device__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") (Device to be used to run the code)

The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

> pip install -r requirements.txt
