from scipy.io import loadmat



data  = loadmat('examples/wrench_example/Wrench.mat')




nodes        = data['nodes']
eleme_coefs  = data['eleme_coefs']

conn_eleme_nodes  = data['conn_eleme_nodes']
conn_eleme_edges  = data['conn_eleme_edges']
conn_edges_nodes  = data['conn_edges_nodes']

nodes_ex  = data['nodes_ex']

edges_neu  = data['edges_neu']
edges_dir  = data['edges_dir']


a = 1



