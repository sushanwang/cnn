The query classification program

## Requirements

- Python 3
- Tensorflow > 1.0.0
- Numpy

## Training

```bash
python server.py
```

## Evaluating

```bash
python server.py --mode eval --checkpoint_dir runs/1489373362/checkpoints/
python server.py --mode eval --data_file "your datafile.txt" --checkpoint_dir runs/1489373362/checkpoints/
```
The first command using the same data as training, just for inspecting what errors are made during training.
Replace the data file path to use you own data
Replace the checkpoint dir with the output from the training.

## Predicting

```bash
python server.py --mode pred --checkpoint_dir runs/1489373362/checkpoints/
```
Replace the checkpoint dir with the output from the training.