
Prerequisites:
software
=============
Python 2.7.9 with the following packages:
    Numpy 1.9.2
    Pyserial
tensorflow 1.9.0
=============

hardware
=============
C51 microcontroller
Stepper motor with four-phase and eight-shot (ours type is "20BY20H04" with driver "DRV8833" )
Camera with macro lens and maximum frame rate of 60fps
Two strip LED light sources whose intensity is adjustable
=============

First step--Feature Extraction:

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
        
2. Do the feature extraction using pre-trained Inception-v3 model for training and validation datasets.
    Firstly, download the pre-trained Inception-v3 model using following command;
    
    $ wget http://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
    
    Then unzip it and get a folder "inception_dec_2015" which includs the model.
    Open a terminal and run the following command;

    $ python getBottleneck_from_inception-v3_pb.py
    
    Every feature vector of images is saved in "bottleneckBullet" folder in the form of "*.txt", the structure of it like as following:
    bottleneckBullet
        |--bad
            |--0_0.jpg.txt
            |--0_1.jpg.txt
            |--...
        |--good
            |--1_0.jpg.txt
            |--1_1.jpg.txt
            |--...
            

Second step--Training the detection model:

1. Training the detection model using the feature vectors generated above by running the following command;
    
    $ python TL_LR_training.py
    
    The training results could be saved in the file "weights.txt".
    
Third step--Loading the HEX file to microcontroller:
1. Compile the C code in the folder of serial-comm_code and generate HEX file;
2. Load the HEX file to microcontroller.
If your stepper motor is not same with us, you may modify the corresponding C code. This step is crucial to the next step.

Fourth step--Testing the detection model:
1. Connect the microcontroller and computer through serial port while connecting the stepper motor and microcontroller.
    Besides, connect the camera and computer.

2. Load the trained well detection model and carry out the detection  for MCS by running the following command;
    
    $python TL_LR_online_test_MCS.py
    
Lastly, the detection results will be outputted in the form of "defective" or "non-defective".
