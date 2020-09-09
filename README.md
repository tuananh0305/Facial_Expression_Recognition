# Facial_Expression_Recognition

This is a kaggle challenge. We want to predict people’s emotion via their face images. The inputs are 48x48 pixel gray-scale images with one face in each image, the outputs are facial expressions which are categorized from one of seven types of emotion (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Sadness, 5: Surprise, 6: Neutral)

# Dataset

The training set consists of 28709 pairs (image & label). The testing set consists of 3589 pairs. All are stored in “csv" file of 2 columns: emotion (number from 0 to 6) and pixels (string contains value of pixel in an image).

![](https://github.com/tuananh0305/Facial_Expression_Recognition/blob/master/imgs/emo1.png)

![](https://github.com/tuananh0305/Facial_Expression_Recognition/blob/master/imgs/emo2.png)

# Part 1: Competition

We construct a model that try to get the accuracy as high as possible. 

Due to the number of "Disgust" images is very small compared with other classes. Therefore it can cause the imbalanced classes problem. In addition, Angry face looks similar to Disgust faces. Thus, we replace "Disgust" label by "Angry" label. Finally, we solve the the facial expression recognition of six emotions: ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

All implementations are done in notebook file.

![](https://github.com/tuananh0305/Facial_Expression_Recognition/blob/master/imgs/emo3.png)

![](https://github.com/tuananh0305/Facial_Expression_Recognition/blob/master/imgs/emo4.png)

# Part 2: Application

We build an application by intergrating the trained model with webcam. We additionally build a three-classes classifier (happy, natural, sad) to get a higher accuracy that will be useful for an application.

All implementation are done in "FER_webcam.py"
