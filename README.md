# AdaptiGuard — Software Repository

## Project Structure
```
adaptiGuard/
├── capture/
│   ├── capture_traffic.py       # Live Wireshark/tcpdump capture (run on PC4)
│   └── simulate_traffic.py      # Dummy traffic generator for offline dev
├── features/
│   ├── extract_features.py      # Extracts flow-level features from .pcap
│   └── feature_definitions.py   # All 18 feature definitions in one place
├── ml/
│   ├── train_model.py           # Train Random Forest + XGBoost
│   ├── evaluate_model.py        # Accuracy, FPR, confusion matrix, graphs
│   └── predict.py               # Load saved model and classify a flow
├── response/
│   └── push_acl.py              # SSH into core switch and push ACL block
├── data/
│   ├── raw/                     # Raw .pcap files go here
│   ├── processed/               # Extracted CSV feature files go here
│   └── models/                  # Saved .pkl model files go here
├── utils/
│   └── logger.py                # Shared logging utility
├── main.py                      # Entry point — runs the full pipeline
└── requirements.txt
```

## How to run (offline, no hardware needed)
```bash
pip install -r requirements.txt
python -m capture.simulate_traffic      # generates dummy_traffic.csv
python -m features.extract_features     # (or skip — simulate already gives features)
python -m ml.train_model                # trains and saves model
python -m ml.evaluate_model             # prints metrics + saves graphs
```

## How to run (with hardware)
1. Complete Phase 1 hardware setup
2. On PC4: `sudo python -m capture.capture_traffic`
3. On PC3: run attack simulations (hping3/Scapy)
4. `python -m features.extract_features data/raw/capture.pcap`
5. `python -m ml.train_model`
6. `python main.py`   ← runs full live pipeline
