
The corresponding paper is:
Implicit active contours driven by Local Binary Fitting Energy.

1. Prepare the dataset.

2. Open matlab, do segmentation using "Local binary Fitting" by running lbf.m in matlab.
    Copy the generated binary photos to "bina_bad" and "bina_good",respectively.

3. Write the binary photos to txt file by running the following command in python:
    
    $python binaimage_to_txt.py
    
4. Copy the txt file to the folder of "Train" as training dataset.
    Then, train the detection model with SVM by running the following command in python:

    $python svmTrain.py
    
    The well trained model will be saved in the form of file "lbf_svm.joblib".
    
5. Test the detection model.
    For the confidentiality reasons, we can't provide the all dataset. But for testing the detection model,
    we can provide five sample images of one MCS. These images are saved in the folder of "test_one_MCS".
    Thus, you can test the detection model by running following command in python:
    
        $python testMCS.py
