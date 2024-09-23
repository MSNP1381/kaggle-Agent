Submissions are evaluated using [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) between the predicted and expected answers.


F1 is calculated as follows:  

F1\=2∗precision∗recallprecision\+recall


where:


precision\=TPTP\+FP


recall\=TPTP\+FN


and:



> True Positive \[TP] \= your prediction is 1, and the ground truth is also 1 \- you predicted a *positive* and that's *true*!  
> 
>  False Positive \[FP] \= your prediction is 1, and the ground truth is 0 \- you predicted a *positive*, and that's *false*.  
> 
>  False Negative \[FN] \= your prediction is 0, and the ground truth is 1 \- you predicted a *negative*, and that's *false*.


Submission File
---------------


For each ID in the test set, you must predict 1 if the tweet is describing a real disaster, and 0 otherwise. The file should contain a header and have the following format:



> id,target  
> 
>  0,0  
> 
>  2,0  
> 
>  3,1  
> 
>  9,0  
> 
>  11,0

