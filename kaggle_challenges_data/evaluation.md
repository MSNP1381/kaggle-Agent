
### Goal

It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable.

### Metric

Submissions are evaluated on [Root-Mean-Squared-Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

### Submission File Format

The file should contain a header and have the following format:

```
Id,SalePrice  
1461,169000.1  
1462,187724.1233  
1463,175221  
etc.
```

You can download an example submission file (sample\_submission.csv) on the [Data page.](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

