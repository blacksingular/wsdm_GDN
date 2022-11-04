# GDN: Alleviating Structrual Distribution Shift in Graph Anomaly Detection
Pytorch Implementation of

Alleviating Structrual Distribution Shift in Graph Anomaly Detection (WSDM 2023)

# Dataset
YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Run `python src/data_process.py` to pre-process the data.

# Reproduce
```sh
python main.py --config ./config/gdn_yelpchi.yml
```

# Acknowledgement
Our code references:
- [CAREGNN](https://github.com/YingtongDou/CARE-GNN)

- [PCGNN](https://github.com/PonderLY/PC-GNN)