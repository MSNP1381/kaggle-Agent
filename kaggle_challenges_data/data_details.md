

What files do I need?
---------------------

You'll need **train.csv**, **test.csv** and **sample\_submission.csv**.

What should I expect the data format to be?
-------------------------------------------

Each sample in the train and test set has the following information:

* The `text` of a tweet
* A `keyword` from that tweet (although this may be blank!)
* The `location` the tweet was sent from (may also be blank)

What am I predicting?
---------------------

You are predicting whether a given tweet is about a real disaster or not. If so, predict a `1`. If not, predict a `0`.

Files
-----

* **train.csv** - the training set
* **test.csv** - the test set
* **sample\_submission.csv** - a sample submission file in the correct format

Columns
-------

* `id` - a unique identifier for each tweet
* `text` - the text of the tweet
* `location` - the location the tweet was sent from (may be blank)
* `keyword` - a particular keyword from the tweet (may be blank)
* `target` - in **train.csv** only, this denotes whether a tweet is about a real disaster (`1`) or not (`0`)
