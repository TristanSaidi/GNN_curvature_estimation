# Scalar Curvature Estimation with Graph Neural Networks
Exploring scalar curvature estimation from graph data with GNNs. 

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data
Run ```generate_data.py``` to generate graphs sampled from the manifold set. The script samples an N-point point cloud from each manifold and creates a k-nearest neighbors graph given said point cloud. 
```bash
python generate_data.py [--output_dir] [--N] [--k]
```

### Train a GNN model
Run ```main.py``` to train a GNN model to predict scalar curvature from subgraphs sampled from the generated graph data.

```bash
python main.py [--exp_name]
```

