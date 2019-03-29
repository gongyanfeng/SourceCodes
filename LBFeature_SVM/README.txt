
1. Prepare the dataset, where the naming convention for images is 0_0.jpg, 0_1.jpg,...,0_*.jpg for defective ones and 1_0.jpg, 1_1.jpg,...,1_*.jpg for non-defective ones.
    The structure of dataset likes as following:
    train_and_val
        |--0_0.jpg
        |--0_1.jpg
        |--...
        |--1_0.jpg
        |--1_1.jpg
        |--...
            
2. Carry out feature extraction to the training dataset by running the following command in python:

    $python LBF_csv.py
    
    All extracted features will be saved in the file of "train_and_val.csv".
    
3. Train the detection model by running the following command in python:

    $python LBF_SVM.py
    
    During the process, SVM classifier will be used to train this model based on the extracted features above,
    and the well trained model will be saved in the form of file "lbf_svm.joblib".
    
4. Test the detection model.
    For the confidentiality reasons, we can't provide the all dataset. But for testing the detection model,
    we can provide five sample images of one MCS. These images are saved in the folder of "test_one_MCS".
    Thus, you can test the detection model by running following command in python:
    
        $python testMCS.py