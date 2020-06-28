from keras.models import load_model

from CNN_Model.resize_image import load_covidx_testing_dataset
from CNN_Model.resnets_utils import convert_to_one_hot

amount_list_test = [100, 150, 20]
X_test_orig, Y_test_orig, classes = load_covidx_testing_dataset(amount_list_test)
X_test = X_test_orig / 255.
Y_test = convert_to_one_hot(Y_test_orig, 3).T

print("number of test examples = " + str(X_test.shape[0]))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

model = load_model("model_result_epochs20_batchsize2_dataset200_200_200.h5")
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
