 # Calcium Imaging Analysis
 



This project takes `.csv` files gathered from Inscopix Data Processing software for processing:

* Syncing traces with externally captured GPIO events.
* General <img src="https://latex.codecogs.com/svg.image?\Delta&space;F/F" /> based statistics.
* Large plotting functinality, 2d/3d scatter, regression, skree, heatmap and correlation matrix.
* Dimensionality reduction with varience filters and principal component analysis.
* Support Vector Machine Learning for classification tasks.
<table>
  <tr>
    <td nowrap><strong>Supported OS</strong></td>
    <td>Linux (list of <a href="./docs/POSIX.md#the-list-of-posix-api-used-in-areg-sdk-including-multicast-message-router" alt="list of POSX API">POSIX API</a>), Windows 7 and higher.</td>
  </tr>
<table>


---
*File handling assumes the following directory structure*:
```bash
./data/
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
    └── Day 1
        ├── *Events.csv
        └── *Traces.csv

```
---
## Config 

1. `DIRS:` - directories for loading and saving data
   ```yaml
   data_dir: '/Users/me/mydata'   # os is internally handled
   save_stats: '~./home'          # save statistical output
   save_figs: '~/graphs'          # can have multiple subdirs 
   save_nn:  '~/svm'              # train/test/eval scores from neural network 
   ```

2. `COLORS:` - {event : color} pairs
   ```yaml
     Event1: 'color'
     Event2: 'color'
     Event3: 'color'
   # from matplotlib.colors API
   ```
   
3. `DECODER:` - Decode GPIO signals into events
   ```yaml
     - Event1: 1, 2, 3, 4
     - Event2: 1, 3, 4
     - Event3: 1, 2, 3
     - Event4: 1, 4
   # see data/data_utils/gpio_data.py for implementation
   ```





