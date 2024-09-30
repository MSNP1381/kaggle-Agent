Goal
----


It is your job to predict if a passenger survived the sinking of the Titanic or not.   
For each in the test set, you must predict a 0 or 1 value for the variable.


Metric
------


Your score is the percentage of passengers you correctly predict. This is known as [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification).


Submission File Format
----------------------


You should submit a csv file with exactly 418 entries **plus** a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.  
  
The file should have exactly 2 columns:


* PassengerId (sorted in any order)
* Survived (contains your binary predictions: 1 for survived, 0 for deceased)



```
PassengerId,Survived  
892,0  
893,1  
894,0  
Etc.
```

You can download an example submission file (gender\_submission.csv) on the [Data page](https://www.kaggle.com/c/titanic/data).

