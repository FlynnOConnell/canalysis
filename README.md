 # Trace Analysis
 

This project takes .csv files gathered from Inscopix Data Processing software for statistical analysis 
and machine learning. 

File handling assumes the following directory structure:

```bash
../data/
├── Animal 1
│   ├── Day 1
│   │   ├── *Events.csv
│   │   └── *Traces.csv
│   └── Day 2
│       ├── *Events.csv
│       └── *Traces.csv
├── Animal 2
│   └── Day 1
│       ├── *Events.csv
│       └── *Traces.csv
└── Animal 3
    ├── Day 1
    │   ├── *Events.csv
    │   └── *Traces.csv
    ├── Day 2
    │   ├── *Events.csv
    │   └── *Traces.csv
    └── Day 3
        ├── *Events.csv
        └── *Traces.csv
```

---

**To Do:** 


**Machine Learning**: Currently the only implemented model for machine learning is _Support Vector Machine_. 
This is due to the relatively small training data available; sessions only contain 4-8 trials of each tastant 
and subsequently 400-600 tastant licks. 

