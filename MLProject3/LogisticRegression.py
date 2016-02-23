__author__ = 'NKHarish'

import re               ## Importing the necessary Packages
import glob
import os
import numpy as np
import scipy
import scipy.io.wavfile
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

def load_fft_data():

    """
    Function to Load the Music Files Data and to extract the first 1000 FFT features

    :return: Returning the obtained FFT 1000 Feature Matrix
    """
    global label_matrix

    file_list=glob.glob(os.path.abspath('FFT_Data/*.txt'))          ## Import the File list in FFT_Data folder
    file_list_length=len(file_list)
    if file_list_length==0:                                         ## Checking if the FFT Features File is present in the FFT_Data Folder

        print("Loading Data from Music Files\n ============================")
        temp_features_matrix=[]
        temp_label_matrix=[]

        for genre in (genre_list):                                  ## Iterating the Feature Extraction for every Genre declared in Genre List

            print("Loading Music Files of "+genre+" Genre")
            file_path=os.path.abspath(os.path.join(input_data_path+genre+"/*.wav"))              ## Obtaining the path of .wav files present in Genre folder
            music_files_list= glob.glob(file_path)                              ## Importing the .wav File list in Genre folder
            music_files_length=len(music_files_list)

            for i in range(music_files_length):                                 ## Iterating the process for all the .wav music files present in the folder

                sample_rate, X = scipy.io.wavfile.read(music_files_list[i])         ## Reading the Music File
                fft_1000_values =abs(scipy.fft(X)[:1000])                           ## Extracting the First 1000 FFT Features from the read music file
                temp_features_matrix.append(fft_1000_values)                        ## Appending the FFT features to a list
                temp_label_matrix.append(genre_list.index(genre))                   ## Appending the Music File Genre to a list

        fft_1000_matrix=np.matrix(temp_features_matrix)
        label_matrix=np.matrix(temp_label_matrix).T
        np.savetxt('FFT_Data/fft_1000_matrix.txt',fft_1000_matrix,'%f')             ## Writing the FFT Feature Matrix to a File, used further
        np.savetxt('FFT_Data/label_matrix.txt',label_matrix,'%d')                   ## Writing the Music File Genre Indexes to a File, used further for calculating the Confusion Matrix

    else:
        print("\n Loading Data from existing FFT Features File \n ----------------------------------------")
        fft_1000_matrix=np.matrix(np.loadtxt('FFT_Data/fft_1000_matrix.txt'))               ## Loading the FFT Features from stored file to matrix
        label_matrix=np.matrix(np.loadtxt('FFT_Data/label_matrix.txt')).T                   ## Loading the Music Genre values from stored file to matrix

    return fft_1000_matrix

def fft_best20_features(fft_features_matrix):

    """
    Function to obtain the Best 20 FFT Features

    :param fft_features_matrix:

    :return: Returning the Best 20 FFT Features Matrix
    """
    all_fft_features_matrix=fft_features_matrix                 ## Creating a copy of FFT Feature Matrix
    fft_features_matrix=np.matrix(np.zeros((no_of_files,no_features)))          ## Initializing the new FFT Feature matrix with (600,20) dimension
    start_index=0
    end_index=0

    for i in range(no_of_labels):               ## Iterating the process of calculating the Best 20 FFT Feature for each Genre

        end_index=start_index+100               ## Initializing the end index
        genre_features_matrix=all_fft_features_matrix[start_index:end_index,:]      ## Extracting all the 1000 FFT Features of a genre
        genre_features_std=genre_features_matrix.std(axis=0)                        ## Calculating the Standard Deviation for all the features in Genre dataset
        all_features_std=all_fft_features_matrix.std(axis=0)                        ## Calculating the Standard Deviation for all the features in complete dataset
        diff_matrix=np.subtract(all_features_std,genre_features_std).T              ## Subtracting tne complete dataset standard deviation and genre dataset standard deviation for all the features
        index=np.matrix(np.arange(1000)).T
        diff_matrix=np.append(diff_matrix,index,axis=1)                             ## Appending the corresponding indexes for the feature standard deviation difference value
        diff_matrix = diff_matrix.view(np.ndarray)
        diff_matrix_sorted=diff_matrix[np.lexsort((diff_matrix[:, 0], ))]           ## Sorting the Difference Matrix
        diff_matrix_best20_indexes=[]
        for i in range(1,21):
            diff_matrix_best20_indexes.append(diff_matrix_sorted[len(diff_matrix_sorted)-i,1])      ## Appending the indexes of 20 features having maximum difference value
        for j in range(no_features):
            fft_features_matrix[start_index:end_index,j]=all_fft_features_matrix[start_index:end_index,int(diff_matrix_best20_indexes[j])]      ## Extracting the Best 20 FFT Features using the Indexes
        start_index=end_index
    return fft_features_matrix

def load_MFCC_data():

    """
    Function to Load the Music Files Data and to extracting the 13 MFCC features

    :return: Returning the MFCC Features Matrix
    """
    global label_matrix

    file_list=glob.glob(os.path.abspath('MFCC_Data/*.txt'))                  ## Import the File list in MFCC_Data folder
    file_list_length=len(file_list)
    if file_list_length==0:                                                   ## Checking if the FFT Features File is present in the MFCC_Data Folder

        print("\n Loading Data from Music Files \n ============================================")
        temp_features_matrix=[]
        temp_label_matrix=[]
        for genre in (genre_list):                                      ## Iterating the Feature Extraction for every Genre declared in Genre List

            print("Loading Music Files of "+genre+" Genre")
            file_path=os.path.abspath(os.path.join(input_data_path+genre+"/*.wav"))              ## Obtaining the path of .wav files present in Genre folder
            music_files_list= glob.glob(file_path)                                               ## Importing the .wav File list in Genre folder
            music_files_length=len(music_files_list)
            for i in range(music_files_length):                                                  ## Iterating the process for all the .wav music files present in the folder

                sample_rate, X = scipy.io.wavfile.read(music_files_list[i])                      ## Reading the Music File
                ceps, mspec, spec = mfcc(X)
                num_ceps = ceps.shape[0]
                mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )   ## Extracting the 13 MFCC Features from the read music file
                temp_features_matrix.append(mfcc_features)                                                    ## Appending the MFCC features to a list
                temp_label_matrix.append(genre_list.index(genre))                                             ## Appending the Music File Genre to a list
        mfcc_13_matrix=np.matrix(temp_features_matrix)
        label_matrix=np.matrix(temp_label_matrix).T
        np.savetxt('MFCC_Data/mfcc_13_matrix.txt',mfcc_13_matrix,'%f')                                        ## Writing the MFCC Feature Matrix to a File, used further
        np.savetxt('MFCC_Data/label_matrix.txt',label_matrix,'%d')                                            ## Writing the Music File Genre Indexes to a File, used further for calculating the Confusion Matrix

    else:
        print("\n Loading Data from existing MFCC Features File \n ----------------------------------------------------")
        mfcc_13_matrix=np.matrix(np.loadtxt('MFCC_Data/mfcc_13_matrix.txt'))                                  ## Loading the MFCC Features from stored file to matrix
        label_matrix=np.matrix(np.loadtxt('MFCC_Data/label_matrix.txt')).T                                    ## Loading the Music Genre values from stored file to matrix

    return mfcc_13_matrix


def kfold_indexes(kvalue):

    """
    Function to calculate indexes for Test and Train data division for all K Folds

    :param kvalue:
    :return:
    """
    global kfold_index_matrix
    kfold_index_matrix=[]
    for i in range(kvalue):                                             ## Iterating the process of calculating the Train and Test Data Indexes for all K folds
        temp_train=[]
        temp_test=[]
        for j in range(no_of_files):                                    ## Iterating over entire file list and classifying the file to either Train or Test dataset
            if abs(j-i)%10!=0:
                temp_train.append(j)
            else:
                temp_test.append(j)
        kfold_index_matrix.append([temp_train,temp_test])
    kfold_index_matrix=np.matrix(kfold_index_matrix)


def get_ifold_matrices(index,features_matrix):

    """
    Function to extract the ith Fold Test, Train Data

    :param index:
    :param features_matrix:
    :return:
    """
    global train_fft_matrix                                             ## Declaring the Train, Test, Test Label matrices as Global
    global test_fft_matrix
    global test_label_matrix

    train_index=kfold_index_matrix[index][0,0]                          ## Obtaining the Kth fold Train and Test Indexes
    test_index=kfold_index_matrix[index][0,1]

    train_fft_matrix=np.zeros((len(train_index),no_features+1))         ## Initializing the Train and Test, Test Label Matrix with zeros
    test_fft_matrix=np.zeros((len(test_index),no_features+1))
    test_label_matrix=np.zeros((len(test_index),1))

    train_fft_matrix[:,0]=1                                             ## Assigning the first column with "1" to accommodate weight Wo
    test_fft_matrix[:,0]=1

    for i in range(len(train_index)):
        train_fft_matrix[i,1:no_features+1]=features_matrix[train_index[i],:]       ## Extracting the Train Data Features from all features matrix

    for j in range(len(test_index)):
        test_fft_matrix[j,1:no_features+1]=features_matrix[test_index[j],:]         ## Extracting the Test Data Features from all features matrix
        test_label_matrix[j,:]=label_matrix[test_index[j],:]                        ## Extracting the Test Data Labels from all Labels matrix


def normalize():

    """
    Function to normalize the ith Fold test, Train Data

    :return:
    """
    for i in range(no_features+1):

        train_fft_matrix[:,i]=train_fft_matrix[:,i]/(train_fft_matrix[:,i].max())           ## Normalizing each column of Train Data Matrix by dividing each value in column by corresponding column maximum value
        test_fft_matrix[:,i]=test_fft_matrix[:,i]/(test_fft_matrix[:,i].max())              ## Normalizing each column of Train Data Matrix by dividing each value in column by corresponding column maximum value


def Train(iteration,local_weight_matrix):

    """
    Function to train the Weight Matrix in Gradient Descent Method

    :param iteration:
    :param local_weight_matrix:
    :return:  Returning the Updated Weight Matrix
    """
    pyx_matrix=logistic_regression(train_fft_matrix,local_weight_matrix)        ## Function call to calculate the P(Y/X,W) for Train Data using Logistic Regression method
    error=delta-pyx_matrix
    new_weight_matrix= local_weight_matrix+(eta/(1+float(iteration)/no_of_iterations))*(np.dot(error,train_fft_matrix)-lam*local_weight_matrix)     ## Expression to calculate the new Weight Matrix

    return new_weight_matrix



def logistic_regression(temp_matrix,pyx_weight_matrix):

    """
    Function to calculate the P(Y/X,W) using Logistic Regression function

    :param temp_matrix:
    :param pyx_weight_matrix:

    :return: Returning the calculated P(Y/X,W) matrix
    """
    temp_matrix=np.dot(pyx_weight_matrix,temp_matrix.T)
    temp_matrix=np.exp(temp_matrix)
    temp_matrix[len(temp_matrix)-1,:]=1
    for i in range(len(temp_matrix[1,:])):
        temp_matrix[:,i]=temp_matrix[:,i]/(np.sum(temp_matrix[:,i]))        ## Expression to calculate the P(Y/X,W) matrix
    return temp_matrix

def classify(classify_weight_matrix):

    """
    Function to classify the Test Matrix

    :param classify_weight_matrix:
    :return:
    """
    global pred_test_label_matrix
    test_pyx_matrix=logistic_regression(test_fft_matrix,classify_weight_matrix)             ## Function call to calculate the P(Y/X,W) for Test Data Matrix
    pred_test_label_matrix=test_pyx_matrix.argmax(axis=0).T                                 ## Obtaining the Test Data Genres based on P(Y/X,W) matrix


def calculate_accuracy():

    """
    Function to calculate the Accuracy & Confusion Matrix

    :return:
    """
    global confusionMatrix
    global accuracy
    confusionMatrix=confusion_matrix(test_label_matrix,pred_test_label_matrix)          ## Building the Confusion Matrix by giving Actual Test Genre matrix and Predicted Test Genre Matrix
    accuracy=(float(np.sum(confusionMatrix.diagonal()))/len(test_label_matrix))*100     ## Calculating the Accuracy value



if __name__=="__main__":                                            ## Main Function, where the program execution flow begins


    global input_data_path   ##Declaring Global Variables
    global genre_list
    global no_of_files
    global no_features
    global no_of_labels
    global eta
    global lam
    global delta
    global no_of_folds
    global no_of_iterations


    input_data_path ='F:/UNM MS/Spring 2015/Machine Learning/Project -3/opihi.cs.uvic.ca/sound/genres/'   ## Declaring the Input Path where Music Files are present
    genre_list=['classical','country','jazz','metal','pop','rock']      ## Declaring the Genre list values
    no_of_files=600             ## Declaring the No. Of Input Files
    no_of_labels=6              ## Declaring the No. Of Output Labels
    no_of_folds=10              ## Declaring the No. Of Folds
    no_of_iterations=300        ## Declaring the No. Of Iterations for training the Weight Matrix
    eta=0.01                    ## Declaring the Learning Rate value
    lam=0.001                   ## Declaring the Lambda value
    delta=np.zeros((no_of_labels,540))      ## Declaring and Initializing the Delta Matrix
    temp_counter=0
    for i in range(0,6):
        for j in range(0,90):
            delta[i,temp_counter]=1
            temp_counter=temp_counter+1


    print("ML Project-3 Logistic Regression")
    print("================================")
    print("Do you wish to use \n 1. FFT Feature Extraction \n 2. MFCC Feature Extraction")

    user_choice=raw_input("Please enter your choice (1 or 2):")         ## Taking the User Choice to use FFT or MFCC Feature Extraction
    user_choice=re.sub(r'\D',"",user_choice).strip()

    if user_choice=="1":                                                ## User Choice "1" refers to Logistic Regression using FFT Feature Extraction

        fft_1000_matrix=load_fft_data()                                 ## Function call to load the Data from Music Files and extract FFT Features
        kfold_indexes(no_of_folds)                                      ## Function call to calculate indexes for Test and Train data division for all K Folds

        print("Do you wish to use \n 1. 1000 FFT Features \n 2. 20 Best FFT Features")

        user_fft_choice=raw_input("Please enter your choice (1 or 2):")     ## Taking the User Choice to use FFT 1000 Features or FFT Best 20 Features.
        user_fft_choice=re.sub(r'\D',"",user_fft_choice).strip()

        if user_fft_choice=="1":                                        ## User choice "1" refers to Logistic Regression using FFT 1000 Feature

            no_features=1000                                            ## Initialize the No of Features as 1000
            weight_matrix=np.zeros((no_of_labels,no_features+1))        ## Initialize Weight Matrix with zeros.

            kfold_fft1000_acc_max=[]
            kfold_fft1000_conf_max=[]
            for i in range(no_of_folds):                                ## Iterating the Process for 10 Folds

                print("Computing "+str(i)+" th Fold Data")
                fold_acc_max =0
                fold_conf_max =0
                get_ifold_matrices(i,fft_1000_matrix)                   ## Function call to extract the ith Fold Test, Train Data
                normalize()                                             ## Function call to normalize the ith Fold test, Train Data
                for iteration in range(no_of_iterations):               ## Iterating the Process of Training the weight matrix,classifying the Test Data,calculating Accuracy for the number of iterations

                    weight_matrix=Train(iteration,weight_matrix)    ## Function call to calculate the Weight Matrix
                    classify(weight_matrix)                                     ## Function call to classify the Test Data
                    calculate_accuracy()                                        ## Function call to calculate Accuracy
                    if(accuracy>=fold_acc_max):
                        fold_acc_max=accuracy                                   ## Retaining the Maximum Accuracy value among the Iteration
                        fold_conf_max=confusionMatrix                           ## Retaining the Confusion Matrix where maximum accuracy is observed

                weight_matrix[:]=0                                              ## Re-initializing the weight matrix to zeros.
                kfold_fft1000_acc_max.append(fold_acc_max)                      ## Appending the List with Maximum Accuracies obtained in every fold
                kfold_fft1000_conf_max.append(fold_conf_max)                    ## Appending the List with Confusion Matrix obtained in every fold

            for i in range(no_of_folds):

                print("Accuracy for "+str(i)+" th Fold : "+str(kfold_fft1000_acc_max[i]))       ## Printing the Accuracies
                print("Confusion Matrix for "+str(i)+" th Fold: \n"+str(kfold_fft1000_conf_max[i]))     ## Printing the Confusion Matrix

            print("Average Accuracy for "+str(no_of_folds)+" Folds: "+str(sum(kfold_fft1000_acc_max)/float(len(kfold_fft1000_acc_max))))      ## Printing the Average Accuracy for 10 Folds


        elif user_fft_choice=="2":                                       ## User choice "1" refers to Logistic Regression using FFT Best 20 Feature

            no_features=20                                               ## Initialize the No of Features as 20
            fft_20_features=fft_best20_features(fft_1000_matrix)         ## Function call to calculate the Best FFT 20 Features per Genre
            weight_matrix=np.zeros((no_of_labels,no_features+1))         ## Initialize Weight Matrix with zeros.

            kfold_fft20_acc_max=[]
            kfold_fft20_conf_max=[]
            for i in range(no_of_folds):                                 ## Iterating the Process for 10 Folds

                print("Computing "+str(i)+" th Fold Data")
                fold_acc_max =0
                fold_conf_max =0
                get_ifold_matrices(i,fft_20_features)                    ## Function call to extract the ith Fold Test, Train Data
                normalize()                                              ## Function call to normalize the ith Fold test, Train Data
                for iteration in range(no_of_iterations):                ## Iterating the Process of Training the weight matrix,classifying the Test Data,calculating Accuracy for the number of iterations

                    weight_matrix=Train(iteration,weight_matrix)    ## Function call to calculate the Weight Matrix
                    classify(weight_matrix)                                     ## Function call to classify the Test Data
                    calculate_accuracy()                                        ## Function call to calculate Accuracy
                    if(accuracy>=fold_acc_max):
                        fold_acc_max=accuracy                                   ## Retaining the Maximum Accuracy value among the Iteration
                        fold_conf_max=confusionMatrix                           ## Retaining the Confusion Matrix where maximum accuracy is observed

                weight_matrix[:]=0                                              ## Re-initializing the weight matrix to zeros.
                kfold_fft20_acc_max.append(fold_acc_max)                        ## Appending the List with Maximum Accuracies obtained in every fold
                kfold_fft20_conf_max.append(fold_conf_max)                      ## Appending the List with Confusion Matrix obtained in every fold

            for i in range(no_of_folds):

                print("Accuracy for "+str(i)+" th Fold : "+str(kfold_fft20_acc_max[i]))         ## Printing the Accuracies
                print("Confusion Matrix for "+str(i)+" th Fold: \n"+str(kfold_fft20_conf_max[i]))    ## Printing the Confusion Matrix

            print("Average Accuracy for "+str(no_of_folds)+" Folds: "+str(sum(kfold_fft20_acc_max)/float(len(kfold_fft20_acc_max))))          ## Printing the Average Accuracy for 10 Folds

        else:
            print("Input is not Valid, Please enter a valid Input (1 or 2)")        ## Displaying Error Message for given Wrong Input

    elif user_choice=="2":                                                    ## User choice "1" refers to Logistic Regression using FFT 1000 Feature

        no_features=13                                                        ## Initialize the No of Features to 13, as MFCC Feature Extraction is used
        no_of_iterations=2000
        mfcc_13_matrix=load_MFCC_data()                                       ## Function call to load the Data from Music Files and extract MFCC Features
        kfold_indexes(no_of_folds)                                            ## Function call to calculate indexes for Test and Train data division for all K Folds

        kfold_mfcc13_acc_max=[]
        kfold_mfcc13_conf_max=[]
        for i in range(no_of_folds):                                          ## Iterating the Process for 10 Folds

            print("Computing "+str(i)+" th Fold Data")
            fold_acc_max =0
            fold_conf_max =0

            get_ifold_matrices(i,mfcc_13_matrix)                              ## Function call to extract the ith Fold Test, Train Data
            normalize()                                                       ## Function call to normalize the ith Fold test, Train Data
            weight_matrix=np.zeros((no_of_labels,no_features+1))              ## Initialize Weight Matrix with zeros.

            for iteration in range(no_of_iterations):                         ## Iterating the Process of Training the weight matrix,classifying the Test Data,calculating Accuracy for the number of iterations

                weight_matrix=Train(iteration,weight_matrix)      ## Function call to calculate the Weight Matrix
                classify(weight_matrix)                                       ## Function call to classify the Test Data
                calculate_accuracy()                                          ## Function call to calculate Accuracy
                if(accuracy>=fold_acc_max):
                    fold_acc_max=accuracy                                     ## Retaining the Maximum Accuracy value among the Iteration
                    fold_conf_max=confusionMatrix                             ## Retaining the Confusion Matrix where maximum accuracy is observed

            weight_matrix[:]=0                                                ## Re-initializing the weight matrix to zeros.
            kfold_mfcc13_acc_max.append(fold_acc_max)                         ## Appending the List with Maximum Accuracies obtained in every fold
            kfold_mfcc13_conf_max.append(fold_conf_max)                       ## Appending the List with Confusion Matrix obtained in every fold

        for i in range(no_of_folds):

            print("Accuracy for "+str(i)+" th Fold : "+str(kfold_mfcc13_acc_max[i]))            ## Printing the Accuracies
            print("Confusion Matrix for "+str(i)+" th Fold: \n"+str(kfold_mfcc13_conf_max[i]))  ## Printing the Confusion Matrix

        print("Average Accuracy for "+str(no_of_folds)+" Folds: "+str(sum(kfold_mfcc13_acc_max)/float(len(kfold_mfcc13_acc_max))))        ## Printing the Average Accuracy for 10 Folds

    else:
        print("Input is not Valid, Please enter a valid Input (1 or 2)")        ## Displaying Error Message for given Wrong Input





