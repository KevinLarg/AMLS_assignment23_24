# AMLS_assignment23_24
Created for Applied Machine Learning Systems module 1, targeting towards a final assignment which is due in early  January.
Two tasks, task A and task B involved
for each task 3 models, SVM, RF and a CNN is tried and had their performance evaluated and compared

Role of Each file:
main.py:
    contains the main working project, designed to present the results;

Dataset folder:
remained empty for pasting dataset files inside. expecting .npz files.

.gitattributes:
    created for git large file storage lfs, for the trained Resnet model;

.gitignore:
    created for not pushing dataset to the github, as required by the assignment;

\A\my_saved_model:
    trained CNN model for task A;

\B\my_saved_model:
    trained Resnet50 CNN model for task B;

\Sundry\testcode_A.ipynb:
    developing models for task A, training cnn and implementing results;

\Sundry\testcode_B.ipynb:
    developing models for task B, training Resnet and implementing results;

requirements.txt:
    provide a list of packages installed;

12/01/24: Note that did try to clone the repository and run the code itself, yet get an error couldn't address:
Cannot parse keras metadata at path A/my_saved_model\keras_metadata.pb: Received error: Error parsing message
This maybe due to the auto-changing of Line Feed done by Git, changing the content of file metadata.pb

Recommend to set the autocrlf to false, maybe helpful
The main.py is working and loading fine in the prepared conda environment.
