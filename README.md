 # Calcium Imaging Analysis
 

This project takes `.csv` files gathered from Inscopix Data Processing software for processing:

* Syncing traces with externally captured GPIO events.
* General <img src="https://latex.codecogs.com/svg.image?\Delta&space;F/F" /> based statistics.
* Large plotting functinality, 2d/3d scatter, regression, skree, heatmap and correlation matrix.
* Dimensionality reduction with varience filters and principal component analysis.
* Support Vector Machine Learning for classification tasks.


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
## Config 

1. 'directory' location of data
   ```python
   datadir = '/Users/me/mydata'
   ```

2. `color_dict` dictionary of event:color pairs
   ```python
   color_dict = {
       'Event1': 'magenta',
       'Event1': 'darkorange',
       'Event1': 'lime',
   }
   # colors from matplotlib.colors API
   ```

