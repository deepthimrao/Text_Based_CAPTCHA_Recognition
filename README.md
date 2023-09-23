# Text_Based_CAPTCHA_Recognition
Advances in deep learning allow us to solve complex AI
problems and achieve good results. One such problem is
text recognition. 

The focus is on finding a way
to capture the text from visual text based Completely Automated Public Turing tests to tell Computers and Humans
Apart (CAPTCHA) using deep learning algorithms. We propose a CRNN (Convolutional Recurrent Neural Network)
based architecture to crack the output of alpha-numeric
5 character visual CAPTCHA. 

Experimented with different CRNN architectures and report results
on different evaluation metrics. Our final CRNN network
achieves an exact measure accuracy of 98.8% and a cosine
similarity of 99.5% on our test data.


Repository information:

The repository contains code for training and inferring of models. It also contains all the trained models and their respective training logs.

preprocess.py = pre-processing code
build_models.py = code for all models (1,2,3,4,5)
train_models.py = code to train models
infer_models.py = code to run inference

In train_models.py, in the main() function, change the input_images_path, ckpt_path, logs_path, model_save_path to yourr desired location and start training

In infer_models.py, in the main() function, change the respective paths to your desired location and perform inference.

captcha_models = contains saved models and checkpoints = This folder has been removed because of size constraint for uploading.

model_1_captcha_1_32_200_50_gray_lstm = Model 1
model_crnn_full_captcha_1_32_200_50_gray_lstm = Model 2
model_crnn_full_captcha_1_32_200_50_gray_gru = Model 3
model_crnn_full_captcha_1_32_200_50_gray_lstm_attn = Model 4
model_crnn_full_captcha_1_32_200_50_gray_gru_attn = Model 5

Refer the report to get detailed explainations for Models 1,2,3,4,5