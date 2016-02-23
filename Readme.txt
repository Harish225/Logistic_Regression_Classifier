=======================================================
ML Project 3 - Implementation of Logistic Regression
=======================================================

1. Project has been developed in Python for 64 bit windows

2. Dowload Pyhton (python-2.7.9 32 bit version) from Official Site

===========================================================
To Install necessary packages required to run the file
=============================================================
Download the .whl file from this link. http://www.lfd.uci.edu/~gohlke/pythonlibs/

1. Download the package NUMPY "numpy‑1.9.2+mkl‑cp27‑none‑win32.whl"
2. Download the package SCIPY "scipy‑0.15.1‑cp27‑none‑win32.whl"
3. Download the package SCIKIT "scikit_learn‑0.16.0‑cp27‑none‑win32.whl"
4. Download the package SKLEARN, RE, GLOB 32 bit versions.

Go to command prompt and install the .whl using this command in python folder.
C:/Python27> python -m pip install .whl filepath

Instructions to run the code from command line
==============================================

1. Download the project folder to your desktop location
2. To run the program without using the saved Features file 
	
	i.	Navigate to \MLProject3\FFT_Data folder and delete all the text files present in that folder.
	ii.	Navigate to \MLProject3\MFCC_Data folder and delete all the text files present in that folder.
	iii. 	Open the \MLProject3\LogisticRegression.py file and replace the Input_data_path variable value(present in Line 273) with the path where Music files are stored on your computer.
	iv. 	Save the changes made to the \MLProject3\LogisticRegression.py file
	v.      Open the command line prompt and navigate to the folder path .....\MLProject3
	vi. 	Type the below command and run the file
			
				python LogisticRegression_FFT.py

3. To run the program using the saved Features file 
	
	i.	Open the command line prompt and navigate to the folder path .....\MLProject3
	ii. 	Type the below command and run the file
			
				python LogisticRegression_FFT.py
				
4. In my case it was C:\Users\NKHarish\PycharmProjects\MLProject3>python LogisticRegression_FFT.py


The Output I have obtained in my cmd is shown below

==============================================================================================================================
		RUNNING THE PROGRAM USING FFT 1000 FEATURES STORED IN LOCAL FILE
==============================================================================================================================

ML Project-3 Logistic Regression
================================
Do you wish to use 
 1. FFT Feature Extraction 
 2. MFCC Feature Extraction
Please enter your choice (1 or 2):
 Loading Data from existing FFT Features File 
 ----------------------------------------
Do you wish to use 
 1. 1000 FFT Features 
 2. 20 Best FFT Features
Please enter your choice (1 or 2):Computing 0 th Fold Data
Computing 1 th Fold Data
Computing 2 th Fold Data
Computing 3 th Fold Data
Computing 4 th Fold Data
Computing 5 th Fold Data
Computing 6 th Fold Data
Computing 7 th Fold Data
Computing 8 th Fold Data
Computing 9 th Fold Data
Accuracy for 0 th Fold : 48.3333333333
Confusion Matrix for 0 th Fold: 
[[10  0  0  0  0  0]
 [ 1  1  3  0  3  2]
 [ 2  0  7  0  1  0]
 [ 4  0  0  0  5  1]
 [ 1  0  0  0  9  0]
 [ 1  3  0  0  4  2]]
Accuracy for 1 th Fold : 58.3333333333
Confusion Matrix for 1 th Fold: 
[[6 0 2 0 0 2]
 [0 2 2 2 0 4]
 [0 1 7 1 0 1]
 [0 0 0 8 1 1]
 [0 1 0 0 7 2]
 [0 2 1 1 1 5]]
Accuracy for 2 th Fold : 46.6666666667
Confusion Matrix for 2 th Fold: 
[[ 5  1  1  1  2  0]
 [ 0  2  1  0  6  1]
 [ 1  0  5  0  3  1]
 [ 0  0  0  3  6  1]
 [ 0  0  0  0 10  0]
 [ 0  1  0  1  5  3]]
Accuracy for 3 th Fold : 50.0
Confusion Matrix for 3 th Fold: 
[[9 0 0 0 1 0]
 [3 2 0 1 4 0]
 [1 0 4 3 2 0]
 [0 0 0 3 7 0]
 [1 0 0 0 9 0]
 [1 0 1 1 4 3]]
Accuracy for 4 th Fold : 45.0
Confusion Matrix for 4 th Fold: 
[[7 0 1 0 1 1]
 [0 2 1 3 0 4]
 [2 0 4 1 0 3]
 [0 0 0 4 1 5]
 [0 0 0 2 4 4]
 [1 2 0 1 0 6]]
Accuracy for 5 th Fold : 48.3333333333
Confusion Matrix for 5 th Fold: 
[[8 0 0 0 1 1]
 [1 1 0 5 2 1]
 [2 0 4 1 2 1]
 [0 0 1 5 4 0]
 [0 1 0 0 9 0]
 [1 1 0 5 1 2]]
Accuracy for 6 th Fold : 55.0
Confusion Matrix for 6 th Fold: 
[[9 0 0 1 0 0]
 [1 2 2 2 3 0]
 [2 0 3 1 0 4]
 [0 0 1 7 1 1]
 [0 0 0 1 9 0]
 [0 0 1 4 2 3]]
Accuracy for 7 th Fold : 55.0
Confusion Matrix for 7 th Fold: 
[[9 0 1 0 0 0]
 [0 4 0 2 1 3]
 [2 0 6 0 0 2]
 [0 0 1 6 3 0]
 [0 0 1 1 6 2]
 [2 0 1 4 1 2]]
Accuracy for 8 th Fold : 50.0
Confusion Matrix for 8 th Fold: 
[[7 1 1 0 0 1]
 [0 4 0 1 3 2]
 [1 1 5 0 0 3]
 [0 0 1 2 4 3]
 [1 0 0 0 7 2]
 [0 2 0 2 1 5]]
Accuracy for 9 th Fold : 43.3333333333
Confusion Matrix for 9 th Fold: 
[[7 0 1 0 1 1]
 [0 0 0 3 2 5]
 [4 0 3 0 0 3]
 [0 0 0 4 2 4]
 [0 0 1 1 8 0]
 [1 1 1 3 0 4]]
Average Accuracy for 10 Folds: 50.0


==============================================================================================================================
		RUNNING THE PROGRAM USING THE BEST 120 FFT FEATURES STORED IN LOCAL FILE
==============================================================================================================================

ML Project-3 Logistic Regression
================================
Do you wish to use 
 1. FFT Feature Extraction 
 2. MFCC Feature Extraction
Please enter your choice (1 or 2):
 Loading Data from existing FFT Features File 
 ----------------------------------------
Do you wish to use 
 1. 1000 FFT Features 
 2. 20 Best FFT Features
Please enter your choice (1 or 2):Computing 0 th Fold Data
Computing 1 th Fold Data
Computing 2 th Fold Data
Computing 3 th Fold Data
Computing 4 th Fold Data
Computing 5 th Fold Data
Computing 6 th Fold Data
Computing 7 th Fold Data
Computing 8 th Fold Data
Computing 9 th Fold Data
Accuracy for 0 th Fold : 63.3333333333
Confusion Matrix for 0 th Fold: 
[[ 8  0  1  0  1  0]
 [ 3  4  0  2  1  0]
 [ 2  3  4  0  1  0]
 [ 0  0  0 10  0  0]
 [ 0  1  1  1  7  0]
 [ 1  1  0  1  2  5]]
Accuracy for 1 th Fold : 63.3333333333
Confusion Matrix for 1 th Fold: 
[[7 3 0 0 0 0]
 [2 6 2 0 0 0]
 [0 4 5 0 0 1]
 [0 0 1 9 0 0]
 [0 2 2 0 6 0]
 [0 4 1 0 0 5]]
Accuracy for 2 th Fold : 56.6666666667
Confusion Matrix for 2 th Fold: 
[[7 1 0 0 0 2]
 [0 5 1 0 3 1]
 [1 2 4 0 2 1]
 [0 0 1 7 2 0]
 [0 3 1 0 5 1]
 [1 2 0 1 0 6]]
Accuracy for 3 th Fold : 61.6666666667
Confusion Matrix for 3 th Fold: 
[[8 0 1 0 1 0]
 [3 5 2 0 0 0]
 [1 2 3 0 4 0]
 [1 1 2 6 0 0]
 [0 0 1 0 9 0]
 [1 1 1 0 1 6]]
Accuracy for 4 th Fold : 58.3333333333
Confusion Matrix for 4 th Fold: 
[[4 0 6 0 0 0]
 [0 6 2 0 2 0]
 [0 2 6 0 1 1]
 [0 0 4 5 0 1]
 [0 0 3 1 6 0]
 [0 0 1 0 1 8]]
Accuracy for 5 th Fold : 56.6666666667
Confusion Matrix for 5 th Fold: 
[[8 0 2 0 0 0]
 [1 4 2 2 0 1]
 [1 2 4 3 0 0]
 [1 0 2 7 0 0]
 [1 1 3 0 4 1]
 [3 0 0 0 0 7]]
Accuracy for 6 th Fold : 63.3333333333
Confusion Matrix for 6 th Fold: 
[[7 0 3 0 0 0]
 [3 6 1 0 0 0]
 [2 2 5 1 0 0]
 [1 1 2 6 0 0]
 [0 2 1 0 7 0]
 [0 0 2 1 0 7]]
Accuracy for 7 th Fold : 58.3333333333
Confusion Matrix for 7 th Fold: 
[[8 0 1 0 1 0]
 [0 6 1 0 3 0]
 [3 1 2 0 3 1]
 [1 2 0 6 0 1]
 [1 2 2 0 5 0]
 [1 0 0 0 1 8]]
Accuracy for 8 th Fold : 70.0
Confusion Matrix for 8 th Fold: 
[[8 0 2 0 0 0]
 [2 6 2 0 0 0]
 [3 0 6 0 1 0]
 [1 0 0 9 0 0]
 [0 1 2 0 7 0]
 [1 2 0 0 1 6]]
Accuracy for 9 th Fold : 61.6666666667
Confusion Matrix for 9 th Fold: 
[[8 1 0 0 1 0]
 [0 7 3 0 0 0]
 [0 4 3 1 2 0]
 [0 2 0 8 0 0]
 [0 5 1 0 4 0]
 [1 2 0 0 0 7]]
Average Accuracy for 10 Folds: 61.3333333333

==============================================================================================================================
		RUNNING THE PROGRAM USING THE MFCC FEATURES STORED IN LOCAL FILE
==============================================================================================================================

ML Project-3 Logistic Regression
================================
Do you wish to use 
 1. FFT Feature Extraction 
 2. MFCC Feature Extraction
Please enter your choice (1 or 2):
 Loading Data from existing MFCC Features File 
 ----------------------------------------------------
Computing 0 th Fold Data
Computing 1 th Fold Data
Computing 2 th Fold Data
Computing 3 th Fold Data
Computing 4 th Fold Data
Computing 5 th Fold Data
Computing 6 th Fold Data
Computing 7 th Fold Data
Computing 8 th Fold Data
Computing 9 th Fold Data
Accuracy for 0 th Fold : 73.3333333333
Confusion Matrix for 0 th Fold: 
[[10  0  0  0  0  0]
 [ 0  6  0  1  1  2]
 [ 3  0  6  1  0  0]
 [ 0  1  0  9  0  0]
 [ 0  1  0  0  8  1]
 [ 0  0  1  4  0  5]]
Accuracy for 1 th Fold : 63.3333333333
Confusion Matrix for 1 th Fold: 
[[8 0 0 1 0 1]
 [0 8 1 1 0 0]
 [2 3 2 2 1 0]
 [1 0 0 9 0 0]
 [1 1 1 0 7 0]
 [1 5 0 0 0 4]]
Accuracy for 2 th Fold : 61.6666666667
Confusion Matrix for 2 th Fold: 
[[8 0 0 0 0 2]
 [1 4 2 1 1 1]
 [3 0 5 1 1 0]
 [1 1 0 8 0 0]
 [0 2 0 0 8 0]
 [0 2 0 4 0 4]]
Accuracy for 3 th Fold : 66.6666666667
Confusion Matrix for 3 th Fold: 
[[8 1 1 0 0 0]
 [0 7 0 0 1 2]
 [2 3 4 0 0 1]
 [0 0 1 9 0 0]
 [0 1 0 0 8 1]
 [0 3 0 1 2 4]]
Accuracy for 4 th Fold : 65.0
Confusion Matrix for 4 th Fold: 
[[ 8  1  0  1  0  0]
 [ 2  4  1  0  2  1]
 [ 1  3  4  1  0  1]
 [ 0  0  0 10  0  0]
 [ 0  0  0  0 10  0]
 [ 0  1  0  5  1  3]]
Accuracy for 5 th Fold : 66.6666666667
Confusion Matrix for 5 th Fold: 
[[ 8  1  1  0  0  0]
 [ 0  7  2  0  0  1]
 [ 4  1  3  1  0  1]
 [ 0  0  0 10  0  0]
 [ 0  0  0  0 10  0]
 [ 1  1  3  3  0  2]]
Accuracy for 6 th Fold : 70.0
Confusion Matrix for 6 th Fold: 
[[ 8  1  1  0  0  0]
 [ 2  4  4  0  0  0]
 [ 3  1  5  0  0  1]
 [ 1  0  0  9  0  0]
 [ 0  0  0  0 10  0]
 [ 1  1  0  2  0  6]]
Accuracy for 7 th Fold : 76.6666666667
Confusion Matrix for 7 th Fold: 
[[ 9  0  1  0  0  0]
 [ 1  7  1  0  0  1]
 [ 1  2  6  0  1  0]
 [ 1  0  0  9  0  0]
 [ 0  0  0  0 10  0]
 [ 0  1  0  4  0  5]]
Accuracy for 8 th Fold : 70.0
Confusion Matrix for 8 th Fold: 
[[10  0  0  0  0  0]
 [ 2  6  1  1  0  0]
 [ 2  1  6  0  1  0]
 [ 0  0  0 10  0  0]
 [ 0  1  0  0  8  1]
 [ 2  1  0  4  1  2]]
Accuracy for 9 th Fold : 75.0
Confusion Matrix for 9 th Fold: 
[[10  0  0  0  0  0]
 [ 1  8  0  1  0  0]
 [ 2  3  5  0  0  0]
 [ 0  0  0 10  0  0]
 [ 0  1  0  1  8  0]
 [ 1  0  2  1  2  4]]
Average Accuracy for 10 Folds: 68.8333333333


