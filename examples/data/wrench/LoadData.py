from pathlib import Path
from scipy.io import loadmat

EXAMPLES_DIR = Path(__file__).resolve().parent.parent

data  = loadmat(str(EXAMPLES_DIR / "wrench_example" / "Wrench.mat"))




nodes        = data['nodes']
eleme_coefs  = data['eleme_coefs']

conn_eleme_nodes  = data['conn_eleme_nodes']
conn_eleme_edges  = data['conn_eleme_edges']
conn_edges_nodes  = data['conn_edges_nodes']

nodes_ex  = data['nodes_ex']

edges_neu  = data['edges_neu']
edges_dir  = data['edges_dir']


a = 1



