# Sentiment Analysis with ML.NET

This sample assumes 1 as toxic and 0 as Not toxic.

It seems similar sentences must be present in both files data.tsv and test.tsv, so accuracy increasesand can be greater than 90%.

So test.tsv file could have a similar sentence, not exactly the same. For example:

|         | data.tsv                       | test.tsv            |
|---------| ------------------------------ | ------------------- |
|Sentence | `0	This is an awesome movie`  | `0	This is awesome` (optional / won´t affect results) |

So the results for sentiments below be something like

```
Model quality metrics evaluation
--------------------------------
Accuracy: 100.00%
Auc: 100.00%
F1Score: 100.00%

Sentiment: This is an awesome movie and you should see it. | Prediction: Not Toxic | Probability: 0.3716927
Sentiment: This is an awesome | Prediction: Not Toxic | Probability: 0.1004808
Sentiment: This is an awesome movie | Prediction: Not Toxic | Probability: 0.06005618
```

If you remove sample sentence `0	This is awesome` from test.tsv, accuracy will still be 100% and results are the same:
```
Sentiment: This is an awesome movie and you should see it. | Prediction: Not Toxic | Probability: 0.3716927
Sentiment: This is an awesome | Prediction: Not Toxic | Probability: 0.1004808
Sentiment: This is an awesome movie | Prediction: Not Toxic | Probability: 0.06005618
```

But if you forget to add sentence in data.tsv, results might fail
```
Sentiment: This is an awesome movie and you should see it. | Prediction: Toxic | Probability: 0.9042763
Sentiment: This is an awesome | Prediction: Toxic | Probability: 0.6573405
Sentiment: This is an awesome movie | Prediction: Toxic | Probability: 0.7090305
```
Adding too many different sentences in data might cause some toxic sentiments to be evaluated as not toxic and vice versa. Not sure what causes this behavior yet.

If data.tsv has a simple sentence, results might be weird since ML.NET is not aware of the context of words. It does not have any idea that awesome is a positive sentiment:
```
|         | data.tsv                       | test.tsv            |
|---------| ------------------------------ | ------------------- |
|Sentence | `0	This is awesome`           |                     |
```

Wierd results since word context is unknown:
```
Accuracy: 95.24%
Auc: 99.09%
F1Score: 95.24%

Sentiment: This is an awesome movie and you should see it. | Prediction: Toxic | Probability: 0.8191079
Sentiment: This is an awesome | Prediction: Not Toxic | Probability: 0.1673261
Sentiment: This is an awesome movie | Prediction: Not Toxic | Probability: 0.3755476
```
