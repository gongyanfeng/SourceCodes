
The corresponding paper is:
Detection and classification of surface defects of gun barrels using computer vision and machine learning.

1. Prepare the dataset, where the naming convention for images is 0_0.jpg, 0_1.jpg,...,0_*.jpg for defective ones and 1_0.jpg, 1_1.jpg,...,1_*.jpg for non-defective ones.
    The structure of dataset likes as following:
    dataset(bullet_photos)
        |--defective(bad)
            |--0_0.jpg
            |--0_1.jpg
            |--...
        |--non-defective(good)
            |--1_0.jpg
            |--1_1.jpg
            |--...

2. Open matlab, do segmentation using Extended-maxima tranform by running Extended_Maxima_Transform.m in matlab.
    Copy the generated binary photos to "train_and_val" which is used for training and validation of the detection model.

3. Carry out feature extraction to the training dataset by running the following command in python:

    $python GLCM_Hist_csv.py
    
    For each binary photo, 6 features of GLCM and 6 features of Histogram will be extracted.
    All extracted features will be saved in the file of "train_and_val.csv".
    
4. Train the detection model by running the following command in python:

    $python GLCM_Hist_SVM.py
    
    Firstly, 5 ones  will be selected out from the 12 features of every image automatically using SequentialFeatureSelector.
    The selected index of features will be saved in the file of "selected_feature_index.txt".
    Then, SVM classifier will be used to train this model, and the well trained model will be saved in the form of file "glcm_hist_svm.joblib".
    
5. Test the detection model.
    For the confidentiality reasons, we can't provide the all dataset. But for testing the detection model,
    we can provide five sample images of one MCS. These images are saved in the folder of "test_one_MCS".
    Thus, you can test the detection model by running following command in python:
    
        $python testMCS.py
        
    The time showed in this process didn't add the part used for segmentation.
    
