import prescient.utils
import pandas as pd 
from sklearn import preprocessing, decomposition
import scanpy as sc
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--input_h5ad", help = "Path to an h5ad file.")
parser.add_argument("--outfile",   default = "growth_rates.pt", required = False, help = "File to be generated.")
parser.add_argument("--birth_gst", default = "hs_birth_msigdb_kegg.csv" , required = False, help = "Path to a csv containing proliferation-associated gene symbols.")
parser.add_argument("--death_gst", default = "hs_death_msigdb_kegg.csv", required = False, help = "Path to a csv containing death-associated gene symbols. Growth rate is computed as birth minus death, so if you want no net change, simply provide the same gene set for both birth and death.")
args = parser.parse_args()
adata = sc.read_h5ad(args.input_h5ad)
expr  = adata.X
try:
    expr = expr.toarray()
except:
    pass
metadata = adata.obs.copy()
assert "timepoint" in metadata.columns, "Metadata must have a 'timepoint' column"
scaler = preprocessing.StandardScaler()
xs = pd.DataFrame(scaler.fit_transform(expr), index = adata.obs_names, columns = adata.var_names)
pca = decomposition.PCA(n_components = 30)
xp_ = pca.fit_transform(xs)
g, g_l=prescient.utils.get_growth_weights(
    xs, xp_, metadata, "timepoint", genes=list(adata.var_names), 
    birth_gst=args.birth_gst, 
    death_gst=args.death_gst, 
    outfile=args.outfile
)
