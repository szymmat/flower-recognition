# flower-recognition
A Python project that allows to train a neural network on any set of labeled images, and then predict a class of an image using trained model. For example, flowers were used. Dataset for training is not uploaded here.
### Training script
Usage: python train.py data_dir <br/>
data_dir is a path to directory with images. Images must be organized in a following way: <br/>
/ <br/>
/class1 <br/>
/class1/image1 <br/>
/class1/image2 <br/>
... <br/>
/class2/ <br/>
/class2/image1 <br/>
/class2/image2 <br/>
etc.
Class names can be chosen freely. The network will be saved in a working directory as checkpoint.pth.
### Predicting script
Usage: python predict.py [--top_k TOP_K] [--category_names CATEGORY_NAMES] path_to_image checkpoint <br/>
positional arguments: <br/>
  *path_to_image*         Path to image to check <br/>
  *checkpoint*            Path to network checkpoint ("./checkpoint.pth" if created with train.py script) <br/>
  options: <br/>
  *--top_k TOP_K*         An integer, number of top classes to return <br/>
  *--category_names CATEGORY_NAMES* Path to JSON file with mapping of categories to real names. Example file is included as cat_to_name.json <br/>
Prints TOP_K most probable classes for an image, along with respective probabilities.
