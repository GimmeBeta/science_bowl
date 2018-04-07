import seaborn as sns

sns.set(color_codes=True)
from utilities import *

# path for the training set directory
stage1_test_path = '/Users/jonathansaragosti/Documents/GitHub/Python/Kaggle/2018 Data Science Bowl/science_bowl/data/stage1_train/'
# Check how many images there are in total in the training set
dataset = DataSet(stage1_test_path, data_set_type='train')

props, image_label_overlay = dataset.subdir[101].get_watershed_props()
print(props)

dir(props[0])
