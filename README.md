 Trace Analysis 
====
This project takes .csv files gathered from Inscopix Data Processing software for statistical analysis 
and machine learning. 

---
**To Do:** 

Clean up:

```python
class Data():
class Traces():
class Events():
```
To make `class Data` contain the most relevant information about the dataset, i.e. traces, cells, 
trial_times and timestamps. 


**Machine Learning**: Currently the only implemented model for machine learning is _Support Vector Machine_. 
This is due to the relatively small training data available; sessions only contain 4-8 trials of each tastant 
and subsequently 400-600 tastant licks. 