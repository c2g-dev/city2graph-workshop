import h3
import osmnx as ox
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
import folium
import contextily as ctx
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import hdbscan
import random
import city2graph as c2g
from torch_geometric.nn import GAE, GATConv

from poi_and_land_uses import (
    classify_poi,
    compute_land_use_ratios,
    save_unmapped_categories,
    save_unmapped_land_use_categories,
)

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
ox.settings.seed = SEED

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMG_DIR = os.path.join(PROJECT_ROOT, "img")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

CLUSTERING_RESULT_PATH = os.path.join(IMG_DIR, "clustering_result.png")
INTERACTIVE_MAP_PATH = os.path.join(DATA_DIR, "interactive_map.html")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "homo_with_embeddings.pt")
PLACE_GEOJSON_PATH = os.path.join(DATA_DIR, "place.geojson")
LAND_USE_GEOJSON_PATH = os.path.join(DATA_DIR, "land_use.geojson")

# ------------------------------------------------------------------ #
# 0. Parameters
# ------------------------------------------------------------------ #
#
# %% [markdown]
# # Parameter Choice Guidelines
# 
# Please follow these steps to choose your parameters safely:
# 
# ## 1. Choose your AREA_NAME
# 
# Choose the **district, borough, or neighbourhood** you want to map.
# 
# - Format the name as `"Neighbourhood, City, Country"`.
#   - `"Camden, London, UK"`
#   - `"Shibuya, Tokyo, Japan"`
# - **Keep it small:** Choose a single district rather than an entire city (e.g. avoid just `"London"`)
#   to complete the data processing and GNN model training in the workshop time.
# - **Check validity:** Take a look at [OpenStreetMap](https://www.openstreetmap.org/). If OSM can find the area boundary from your search term, it can be used here.
# 
# ## 2. Generate Hexes and Check the Count
# 
# Keep `H3_RESOLUTION = 9` (the default) for your first try. Run **Step 1** below and look at the printed output to see how many **H3 hexagons** are generated for your area.
# 
# ## 3. Check Expected Runtime and Adjust (Optional)
# 
# Check your hexagon count against the expected total runtime below (indicative wall-clock times on a CPU-only laptop):
# 
# | Hex count (approx) | Expected runtime |
# |:-------------------|:-----------------|
# | < 100              | < 1 min          |
# | ~200 - 500         | ~1-3 min         |
# | ~1,000             | ~5-15 min        |
# | ~7,000             | ~30-90 min       |
# | ~20,000+           | several hours+   |
# 
# Note: The majority of runtime comes from Step 6 (metapath computation via Dijkstra). For a workshop demonstration, keeping the hex count under 1,000 is recommended.
# 
# If the generated hexagons are too coarse (too large) for your area, you can change the `H3_RESOLUTION` to 10 (finer detail) and re-run Step 1. 
# Please note that Resolution 10 yields roughly 7x more hexagons for the same area compared to Resolution 9, which will correspondingly increase the runtime.
#
# ---
# 
# **FIXED parameters** (do not change for this workshop):
# - `WALKING_SPEED_MPS` = 4.8 km/h  *(standard walking speed)*
# - `THRESHOLD_SEC`     = 15 min     *(15-minute walk reachability)*
# 
# %%

AREA_NAME = "Shibuya, Tokyo, Japan"    # ← change to your area (see AREA_NAME guide above)
WALKING_SPEED_MPS = 4.8 / 3.6  # 4.8 km/h → m/s  [FIXED]
THRESHOLD_SEC = 15 * 60.0        # 15 minutes in seconds  [FIXED]
H3_RESOLUTION = 9              # Start from 9. If too coarse, try finer ones like 10
USE_CONTIGUITY_GRAPH_ONLY = True  # Set True for mega-cities to skip 15-min metapath routing


# %% [markdown]
# # Step 1: Define Area & Generate H3 Hexagons
# In this step, we fetch the boundary polygon of your chosen `AREA_NAME` from Overture Maps.
# The script automatically determines the best projected coordinate system (EPSG code) for your area.
# Then, we fill this polygon with H3 hexagonal cells at the specified `H3_RESOLUTION`.
# These hexagons will act as the "nodes" in our graph, forming a uniform grid over our area of interest.
# %%
boundary_gdf = c2g.get_boundaries(AREA_NAME)

# Automatically determine the best projected CRS (UTM zone) for the area
EPSG = boundary_gdf.estimate_utm_crs().to_epsg()
print(f"Automatically determined EPSG: {EPSG}")

# Combine geometries into a single (Multi)Polygon
boundary_polygon = boundary_gdf.geometry.union_all()

# Fill boundary with H3 cells
h3_indices = set()
if boundary_polygon.geom_type == "MultiPolygon":
    for poly in boundary_polygon.geoms:
        geo = {
            "type": "Polygon",
            "coordinates": [list(poly.exterior.coords)],
        }
        h3_indices |= set(h3.geo_to_cells(geo, res=H3_RESOLUTION))
else:
    geo = {
        "type": "Polygon",
        "coordinates": [list(boundary_polygon.exterior.coords)],
    }
    h3_indices = set(h3.geo_to_cells(geo, res=H3_RESOLUTION))

h3_indices = sorted(h3_indices)
print(f"H3 hexagons (res {H3_RESOLUTION}): {len(h3_indices)}")

# Build hex polygons + centroids
hex_records = []
for h3_id in h3_indices:
    boundary_coords = h3.cell_to_boundary(h3_id)  # [(lat, lng), ...]
    polygon = Polygon([(lng, lat) for lat, lng in boundary_coords])
    lat, lng = h3.cell_to_latlng(h3_id)
    hex_records.append({
        "h3_index": h3_id,
        "geometry": Point(lng, lat),   # centroid as node geometry
        "hex_polygon": polygon,        # keep polygon for reference
    })

hex_gdf = (
    gpd.GeoDataFrame(hex_records, crs="EPSG:4326")
    .set_index("h3_index")
    .to_crs(epsg=EPSG)
)
# Make sure to project the hex_polygon as well for accurate spatial joins
hex_gdf["hex_polygon"] = gpd.GeoSeries(hex_gdf["hex_polygon"], crs="EPSG:4326").to_crs(epsg=EPSG)

print(f"Hex nodes (centroids): {len(hex_gdf)}")

# %% [markdown]
# # Step 2: Fetch Spatial Context (POIs & Landuse)
# A graph needs meaningful node features. Here we enrich our hexagon nodes with real-world data from Overture Maps:
# - **POIs (Points of Interest)**: We count how many shops, cafes, offices, etc., are inside each hexagon.
# - **Landuse**: We determine whether the hex is primarily residential, commercial, industrial, etc.
# 
# Note: Depending on the area size and internet speed, fetching Overture data may take 1-2 minutes.
# %%

print("Fetching Overture Maps data for Places and Land Use...")
overture_data = c2g.load_overture_data(
    area=boundary_polygon,
    types=["place", "land_use"],
    output_dir=str(DATA_DIR),
    save_to_file=True,
    return_data=True,
)

print(f"Saved place data to '{PLACE_GEOJSON_PATH}'")
print(f"Saved land use data to '{LAND_USE_GEOJSON_PATH}'")

# 2.1 Process POIs (Places)
# Overture's "place" type covers amenities, shops, tourism, etc.

pois_gdf = overture_data["place"].to_crs(epsg=EPSG)
pois_gdf["categories"] = pois_gdf.get("categories", None) # Safe access
pois_gdf["functional_class"] = pois_gdf["categories"].apply(classify_poi)

save_unmapped_categories()

# Use centroids for polygon POIs to perform point-in-polygon join
pois_gdf["geometry"] = pois_gdf.geometry.centroid

# Perform Spatial Join: POIs within Hex Polygons
hex_poly_gdf = hex_gdf.set_geometry("hex_polygon")
joined_pois = gpd.sjoin(pois_gdf, hex_poly_gdf, how="inner", predicate="within")

# Count total POIs per hex cell
poi_counts = joined_pois.groupby("h3_index").size().rename("poi_count")
hex_gdf["poi_count"] = hex_gdf.index.map(poi_counts).fillna(0).astype(int)

# Count POIs by functional class per hex cell
if "functional_class" in joined_pois:
    poi_counts_by_class = joined_pois.groupby(["h3_index", "functional_class"]).size().unstack(fill_value=0)
    poi_counts_by_class = poi_counts_by_class.reindex(hex_gdf.index).fillna(0).astype(int)
    hex_gdf = hex_gdf.join(poi_counts_by_class)


# 2.2 Process Landuse
landuse_gdf = overture_data["land_use"].to_crs(epsg=EPSG)

# Determine the main landuse category. In Overture, `subtype` or `class` are useful classifications.
if "subtype" in landuse_gdf.columns and "class" in landuse_gdf.columns:
    landuse_gdf["landuse_category"] = landuse_gdf["subtype"].fillna(landuse_gdf["class"])
elif "subtype" in landuse_gdf.columns:
    landuse_gdf["landuse_category"] = landuse_gdf["subtype"]
elif "class" in landuse_gdf.columns:
    landuse_gdf["landuse_category"] = landuse_gdf["class"]
else:
    landuse_gdf["landuse_category"] = "unknown"

landuse_ratio_cols = [
    column for column in hex_gdf.columns
    if isinstance(column, str) and column.startswith("land_use_") and column.endswith("_ratio")
]
if landuse_ratio_cols:
    hex_gdf = hex_gdf.drop(columns=landuse_ratio_cols)

landuse_ratios, dominant_landuse = compute_land_use_ratios(hex_gdf, landuse_gdf)
save_unmapped_land_use_categories()
hex_gdf["landuse"] = dominant_landuse
if not landuse_ratios.empty:
    hex_gdf[landuse_ratios.columns] = landuse_ratios

print("Added POI counts and landuse attributes. Sample:")
cols_to_print = ["poi_count", "landuse"]
if "functional_class" in joined_pois:
    cols_to_print += list(poi_counts_by_class.columns[:3]) # show a few categorical POIs
if not landuse_ratios.empty:
    cols_to_print += list(landuse_ratios.columns[:3])
print(hex_gdf[cols_to_print].head())

if USE_CONTIGUITY_GRAPH_ONLY:
    # %% [markdown]
    # # Step 3: Compute Hexagon Contiguity (H3 Neighbors)
    # Instead of routing on the street network, we build links directly between adjacent
    # H3 hexagons. This is much faster and suitable for mega-cities.
    # %%
    print("USE_CONTIGUITY_GRAPH_ONLY=True: using contiguity graph (skipping 15-min metapaths).")
    print("Calculating H3 contiguity edges...")
    _, walk_edges = c2g.contiguity_graph(
        gdf=hex_gdf.set_geometry("hex_polygon"),
        contiguity="queen",
        distance_metric="euclidean",
    )

    # Convert Euclidean edge distance to travel time.
    if not walk_edges.empty:
        walk_edges["travel_time_sec"] = walk_edges["weight"] / WALKING_SPEED_MPS

    print(f"H3 contiguity edges (hex–hex): {len(walk_edges)}")
else:
    # %% [markdown]
    # # Step 3: Fetch Street Network
    # To measure walking times, we need a walkable street network.
    # Below, we download the local walking network from OpenStreetMap via OSMnx.
    # We then calculate the `travel_time_sec` for every street segment based on our expected `WALKING_SPEED_MPS`.
    #
    # Note: Fetching the street graph from OSMnx can take roughly 1 to 5 minutes depending on area size.
    # %%
    # ── Option A: OSMnx ──────────────────────────────────────────────────
    print("Fetching street network from OSM...")
    G = ox.graph_from_place(AREA_NAME, network_type="walk")

    # Convert NetworkX → node / edge GeoDataFrames
    street_nodes, street_edges = c2g.nx_to_gdf(G)

    # Ensure same CRS
    street_nodes = street_nodes.to_crs(epsg=EPSG)
    street_edges = street_edges.to_crs(epsg=EPSG)

    # %% [markdown]
    # *(Optional Advanced)*: Instead of OSMnx, you can fetch street segments directly from Overture Maps.
    # If you wish to try this, you can comment out Option A above and uncomment Option B below.
    # %%
    # # ── Option B: Overture Maps segments ────────────────────────────────
    # print("Fetching street segments from Overture Maps...")
    # overture_streets = c2g.load_overture_data(
    #     area=boundary_polygon,
    #     types=["segment", "connector"],
    # )
    #
    # # Process segments: split at connectors, compute barriers & lengths
    # processed_segments = c2g.process_overture_segments(
    #     segments_gdf=overture_streets["segment"].to_crs(epsg=EPSG),
    #     connectors_gdf=overture_streets["connector"].to_crs(epsg=EPSG),
    #     threshold=1.0,
    # )
    #
    # # Build a NetworkX graph from the processed edges, then extract
    # # node / edge GeoDataFrames so the format matches Option A.
    # G_overture = c2g.gdf_to_nx(edges=processed_segments, directed=False)
    # street_nodes, street_edges = c2g.nx_to_gdf(G_overture)

    # Add travel time on street edges (length in metres after projection)
    street_edges["travel_time_sec"] = street_edges.length / WALKING_SPEED_MPS
    print(f"Street nodes: {len(street_nodes)},  Street edges: {len(street_edges)}")

    # %% [markdown]
    # # Step 4: Build Heterogeneous Graph Structure
    # Now we start building a multi-modal "heterogeneous graph".
    # It will have two types of nodes:
    # 1. `hex`: The H3 hexagons we created in Step 1.
    # 2. `street_connector`: The intersections of the street network from Step 3.
    # %%

    hetero_nodes = {
        "hex": hex_gdf,
        "street_connector": street_nodes,
    }

    hetero_edges = {
        ("street_connector", "is_connected_to", "street_connector"): street_edges,
    }

    # %% [markdown]
    # # Step 5: Connect Hexagons to the Street Network
    # To simulate a person walking from a given hexagon, we must connect each hexagon centroid to its nearest street intersection.
    # We call these "bridge edges", which allow traversal between the abstract hex grid and the physical street network.
    # %%

    bridge_key = ("hex", "is_nearby", "street_connector")

    _, bridged_edges = c2g.bridge_nodes(
        nodes_dict=hetero_nodes,
        source_node_types=["hex"],
        target_node_types=["street_connector"],
        k=1,
    )

    hetero_edges.update(bridged_edges)

    # Add travel time on bridged edges
    hetero_edges[bridge_key]["travel_time_sec"] = (
        bridged_edges[bridge_key].length / WALKING_SPEED_MPS
    )
    print(f"Bridge edges (hex → street): {len(hetero_edges[bridge_key])}")

    # %% [markdown]
    # # Step 6: Compute 15-Minute Walking "Metapaths"
    # In this step, for every hexagon, we simulate walking outwards along the street network for up to 15 minutes.
    # Whenever we reach another hexagon within that time, we create a direct "metapath" edge linking the two hexagons, labeled `15_min_walk`.
    #
    # Workshop Note: This is a computationally intensive step (running Dijkstra's algorithm across the graph).
    # - **Small/Medium areas**: Takes 1~5 minutes.
    # - **Large areas**: Might take 15+ minutes.
    # %%

    metapath_key = ("hex", "15_min_walk", "hex")

    print("Calculating metapaths...")
    hetero_nodes, hetero_edges = c2g.add_metapaths_by_weight(
        nodes=hetero_nodes,
        edges=hetero_edges,
        weight="travel_time_sec",
        threshold=THRESHOLD_SEC,
        new_relation_name="15_min_walk",
        endpoint_type="hex",
        edge_types=[
            bridge_key,
            ("street_connector", "is_connected_to", "street_connector"),
        ],
        directed=False,
    )

    walk_edges = hetero_edges[metapath_key]
    print(f"15-min walk metapath edges (hex–hex): {len(walk_edges)}")

# %% [markdown]
# # Step 7: Preprocessing Features for Deep Learning
# Machine learning models train much more effectively on scaled data. In this step, we:
# 1. **Scale Land-Use Occupancy Ratios** so each feature reflects the share of each hexagon occupied by a land-use class.
# 2. **Log-Transform & Standard-Scale** the POI counts, mitigating the effect of extreme outliers.
# 3. **Min-Max Scale** the travel times on our edges to be tightly bounded between 0 and 1.
# 
# Finally, we convert our data to a PyTorch Geometric (PyG) homogenous `Data` object.
# %%

print("Scaling node attributes...")

# Collect POI-derived feature columns
if "functional_class" in joined_pois:
    poi_feature_cols = list(poi_counts_by_class.columns)

# Log1p + z-score scaling for counts
count_scaler = StandardScaler()
scaled_poi_features = count_scaler.fit_transform(
    np.log1p(hex_gdf[poi_feature_cols]).to_numpy(dtype=float)
)
hex_gdf[poi_feature_cols] = scaled_poi_features

# Scale land-use occupancy ratios so all node attributes go through the same pipeline.
landuse_feature_cols = [
    column for column in hex_gdf.columns
    if isinstance(column, str) and column.startswith("land_use_") and column.endswith("_ratio")
]
if landuse_feature_cols:
    landuse_scaler = StandardScaler()
    hex_gdf[landuse_feature_cols] = landuse_scaler.fit_transform(
        hex_gdf[landuse_feature_cols].astype(float).to_numpy()
    )

all_feature_cols = poi_feature_cols + landuse_feature_cols

print("Scaling edge attributes...")
edge_scaler = MinMaxScaler()
walk_edges["travel_time_sec"] = edge_scaler.fit_transform(walk_edges[["travel_time_sec"]])

print("Converting to PyG homogeneous graph...")
homo_data = c2g.gdf_to_pyg(
    nodes=hex_gdf,
    edges=walk_edges,
    node_feature_cols=all_feature_cols,
    edge_feature_cols=["travel_time_sec"],
)

print(homo_data)

# %% [markdown]
# # Step 8: Train Graph Neural Network (GAE)
# We train an unsupervised Graph Autoencoder to learn a dense vector "embedding" for each hexagon.
# Using Graph Attention Networks (GAT), the model learns an embedding based not just on the hexagon's own POI/landuse features, but also by "attending" to the features of other hexagons reachable within a 15-minute walk.
# 
# You can monitor the loss moving downward over the 500 epochs.
# %%

print("Setting up Graph Autoencoder...")

# For unsupervised learning on the full graph
data = homo_data.clone()


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, edge_dim=1):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_attr=edge_attr)

in_channels = homo_data.num_node_features
hidden_channels = 16
out_channels = 8

model = GAE(GATEncoder(in_channels, hidden_channels, out_channels, edge_dim=1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data = data.to(device)
# travel_time_sec is in data.edge_attr of shape [num_edges, 1]. Use it as edge_attr.
edge_attr = data.edge_attr.view(-1, 1).float()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index, edge_attr)
    
    # Reconstruct the adjacency matrix using all edges
    loss = model.recon_loss(z, data.edge_index)
    
    loss.backward()
    optimizer.step()
    return float(loss)

print("Training GAE on full graph...")
epochs = 500
for epoch in range(1, epochs + 1):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
# Extract and attach learned embeddings
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index, edge_attr)
    
homo_data.embeddings = z.cpu()
print(f"Learned embeddings shape: {homo_data.embeddings.shape}")

# Attach embeddings as node features for PyG to GDF conversion
z_np = homo_data.embeddings.numpy()
num_emb_cols = z_np.shape[1]
emb_cols = []
for i in range(num_emb_cols):
    col_name = f'embedding_{i}'
    setattr(homo_data, col_name, homo_data.embeddings[:, i])
    emb_cols.append(col_name)

# %% [markdown]
# # Step 9: Clustering Urban Neighborhoods
# We use the learned embeddings to group hexagons with similar urban functions and accessibility properties.
# We evaluate:
# - **HDBSCAN**: Identifies dense clusters of varying shapes, while gracefully ignoring "noise" outliers (labeled as -1).
# - **K-Means**: Forces every hexagon into 1 of K clusters.
#
# After clustering, we plot our map as a static image.
# %%
print("Converting PyG data back to GeoDataFrame...")
nodes_gdf, _ = c2g.pyg_to_gdf(homo_data, additional_node_cols=emb_cols)

# Restore original hex polygons for visualization
# nodes_gdf geometry might be points (centroids), so we overwrite with polygons
nodes_gdf = nodes_gdf.set_geometry(hex_gdf["hex_polygon"].values)

# Extract embedding matrix and normalize
Z = nodes_gdf[emb_cols].values
Z_norm = normalize(Z, norm='l2')


def build_cluster_color_map(labels, cmap_name="turbo", noise_label=None, noise_color="#333333"):
    unique_labels = sorted(set(labels))
    mapped_labels = [label for label in unique_labels if label != noise_label]

    palette = sns.color_palette(cmap_name, len(mapped_labels)) if mapped_labels else []
    color_map = {
        label: mcolors.to_hex(color)
        for label, color in zip(mapped_labels, palette)
    }

    if noise_label is not None and noise_label in unique_labels:
        color_map[noise_label] = noise_color

    return color_map

print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=15)
labels_hdbscan = clusterer.fit_predict(Z_norm)
nodes_gdf['cluster_hdbscan'] = labels_hdbscan
hdbscan_color_map = build_cluster_color_map(labels_hdbscan, noise_label=-1)

print("Running K-Means clustering (k=2 to 10)...")
best_k = -1
best_score = -1
best_labels_kmeans = None

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(Z_norm)
    score = silhouette_score(Z_norm, labels)
    print(f"  k={k}: Silhouette Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
        best_labels_kmeans = labels

print(f"Selected K-Means with k={best_k} (Score: {best_score:.4f})")
nodes_gdf['cluster_kmeans'] = best_labels_kmeans
kmeans_color_map = build_cluster_color_map(best_labels_kmeans)

print("Visualizing clustering results...")
plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='#0f0f0f')
ax.set_facecolor('#0f0f0f')

# Basemap: hex polygons
hex_poly_gdf = hex_gdf.set_geometry("hex_polygon")
hex_poly_gdf.plot(ax=ax, facecolor='#161616', edgecolor='#333333', linewidth=0.6, alpha=0.5)

# Add Contextily basemap
ctx.add_basemap(
    ax,
    crs=hex_poly_gdf.crs.to_string(),
    source=ctx.providers.CartoDB.DarkMatter,
    alpha=1,
    attribution=""
)

# Edges with colors (15-min walk metapaths)
original_travel_time = edge_scaler.inverse_transform(walk_edges[["travel_time_sec"]])
walk_edges_plot = walk_edges.assign(travel_time_original=original_travel_time)
norm = mcolors.Normalize(vmin=0, vmax=900)
walk_edges_plot.plot(
    ax=ax, 
    column='travel_time_original', 
    cmap='RdYlBu', 
    norm=norm, 
    linewidth=1.0, 
    alpha=0.9
)

# Generate colors for HDBSCAN, treating -1 as noise (dark grey)
colors_hdbscan = [hdbscan_color_map[label] for label in labels_hdbscan]

c2g.plot_graph(
    nodes=nodes_gdf,
    ax=ax,
    node_color=colors_hdbscan,
    node_edgecolor='#ffffff',
    node_alpha=0.85,
    linewidth=0.2,
    bgcolor='#0f0f0f'
)
ax.set_title("HDBSCAN Clustering\n(min_cluster_size=15, min_samples=15)", fontsize=16, color='white')
ax.set_axis_off()

fig.text(0.5, 0.02, '© OpenStreetMap contributors', ha='center', fontsize=10, color='#cccccc')

plt.tight_layout()
plt.savefig(CLUSTERING_RESULT_PATH, dpi=300, facecolor='#0f0f0f', bbox_inches='tight')
print(f"Saved clustering results to '{CLUSTERING_RESULT_PATH}'")

# %% [markdown]
# # Step 10: Generate Interactive Web Map
# In this step, using `folium`, we export all our geographic layers (Metapaths, POIs, Hex Clusters) into an interactive `interactive_map.html` file.
# 
# Once the cell finishes, open `interactive_map.html` in your web browser to explore your results.
# %%

print("\nGenerating interactive map...")

# Ensure layers are in EPSG:4326 for folium
m_nodes = nodes_gdf.to_crs(epsg=4326)
m_walk = walk_edges.to_crs(epsg=4326)
m_pois = pois_gdf.to_crs(epsg=4326)

# Create base map centered around the area
bounds = m_nodes.total_bounds
center_lat_lon = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
m = folium.Map(location=center_lat_lon, zoom_start=15, tiles="CartoDB dark_matter")

'''
# 1. Add 15 Min Walk layer
m_walk_clean = m_walk[["geometry"]].copy()
m_walk_clean.explore(
    color="gray",
    name="15 Min Walk",
    m=m,
    show=False,
    style_kwds={"weight": 1.0, "opacity": 0.4}
)

# 2. Add POIs layer
m_pois_clean = m_pois[["functional_class", "geometry"]].copy()
m_pois_clean.explore(
    column="functional_class",
    categorical=True,
    name="POIs",
    m=m,
    marker_type="circle_marker",
    marker_kwds={"radius": 5, "fill": True},
    tooltip="functional_class",
    legend=False,  # Prevent legend overlap with clusters
    style_kwds={"fillOpacity": 0.8, "weight": 0}
)
'''

# 3. Add HDBSCAN clusters layer (on top of streets and POIs, but with opacity)
m_nodes_clean = m_nodes[["cluster_hdbscan", "geometry"]].copy()
# Reset index if we want it in tooltip (it's h3_index)
m_nodes_clean = m_nodes_clean.reset_index()
m_nodes_clean["cluster_color"] = m_nodes_clean["cluster_hdbscan"].map(hdbscan_color_map)

hdbscan_layer = folium.FeatureGroup(name="HDBSCAN Clusters")
folium.GeoJson(
    data=m_nodes_clean[["h3_index", "cluster_hdbscan", "cluster_color", "geometry"]].to_json(),
    style_function=lambda feature: {
        "fillColor": feature["properties"]["cluster_color"],
        "color": feature["properties"]["cluster_color"],
        "weight": 1,
        "fillOpacity": 0.4,
    },
    tooltip=folium.GeoJsonTooltip(fields=["h3_index", "cluster_hdbscan"]),
).add_to(hdbscan_layer)
hdbscan_layer.add_to(m)

# 4. Add K-Means clusters layer
m_nodes_kmeans = m_nodes[["cluster_kmeans", "geometry"]].copy()
m_nodes_kmeans = m_nodes_kmeans.reset_index()
m_nodes_kmeans["cluster_color"] = m_nodes_kmeans["cluster_kmeans"].map(kmeans_color_map)

kmeans_layer = folium.FeatureGroup(name="K-Means Clusters", show=False)
folium.GeoJson(
    data=m_nodes_kmeans[["h3_index", "cluster_kmeans", "cluster_color", "geometry"]].to_json(),
    style_function=lambda feature: {
        "fillColor": feature["properties"]["cluster_color"],
        "color": feature["properties"]["cluster_color"],
        "weight": 1,
        "fillOpacity": 0.4,
    },
    tooltip=folium.GeoJsonTooltip(fields=["h3_index", "cluster_kmeans"]),
).add_to(kmeans_layer)
kmeans_layer.add_to(m)

# Add Layer Control
folium.LayerControl().add_to(m)

# Save to HTML
map_filename = INTERACTIVE_MAP_PATH
m.save(map_filename)
print(f"Saved interactive map to '{map_filename}'")

# If you want to save the graph with embeddings, uncomment the following line:
# torch.save(homo_data, EMBEDDINGS_PATH)
