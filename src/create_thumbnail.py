import os
import random
import warnings
import h3
import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch

import city2graph as c2g
from torch_geometric.nn import GAE, GATConv
import torch.nn.functional as F

from src.poi_and_land_uses import (
    classify_poi,
    compute_land_use_ratios,
    save_unmapped_categories,
    save_unmapped_land_use_categories,
)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMG_DIR = os.path.join(PROJECT_ROOT, "img")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Parameters
# ------------------------------------------------------------------ #
H3_RESOLUTION = 9              # ~66 m edge length — urban-block scale
WALKING_SPEED_MPS = 4.8 / 3.6  # 4.8 km/h → m/s  [FIXED]
THRESHOLD_SEC = 15 * 60.0        # 15 minutes in seconds  [FIXED]
AREA_NAME = "Shibuya, Tokyo, Japan"    # ← change to your area
EPSG = 6677                     # ← must match AREA_NAME

def process_city(area_name, epsg, h3_res=10, mode="bridges", do_metapaths=False, do_clustering=False):
    print(f"\n--- Processing {area_name} ({mode}) ---")

    # 1. Boundary
    boundary_gdf = c2g.get_boundaries(area_name)
    boundary_polygon = boundary_gdf.geometry.union_all()

    # 2. H3 hexes
    h3_indices = set()
    if boundary_polygon.geom_type == "MultiPolygon":
        for poly in boundary_polygon.geoms:
            geo = {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
            h3_indices |= set(h3.geo_to_cells(geo, res=h3_res))
    else:
        geo = {"type": "Polygon", "coordinates": [list(boundary_polygon.exterior.coords)]}
        h3_indices = set(h3.geo_to_cells(geo, res=h3_res))
    
    h3_indices = sorted(h3_indices)
    
    hex_records = []
    for h3_id in h3_indices:
        boundary_coords = h3.cell_to_boundary(h3_id)
        polygon = Polygon([(lng, lat) for lat, lng in boundary_coords])
        lat, lng = h3.cell_to_latlng(h3_id)
        hex_records.append({
            "h3_index": h3_id,
            "geometry": Point(lng, lat),
            "hex_polygon": polygon,
        })
    hex_gdf = gpd.GeoDataFrame(hex_records, crs="EPSG:4326").set_index("h3_index").to_crs(epsg=epsg)
    hex_gdf["hex_polygon"] = gpd.GeoSeries(hex_gdf["hex_polygon"], crs="EPSG:4326").to_crs(epsg=epsg)

    # 3. Streets
    G = ox.graph_from_place(area_name, network_type="walk")
    street_nodes, street_edges = c2g.nx_to_gdf(G)
    street_nodes = street_nodes.to_crs(epsg=epsg)
    street_edges = street_edges.to_crs(epsg=epsg)
    street_edges["travel_time_sec"] = street_edges.length / WALKING_SPEED_MPS

    hetero_nodes = {
        "hex": hex_gdf,
        "street_connector": street_nodes,
    }
    hetero_edges = {
        ("street_connector", "is_connected_to", "street_connector"): street_edges,
    }

    # 4. Bridges
    bridge_key = ("hex", "is_nearby", "street_connector")
    _, bridged_edges = c2g.bridge_nodes(
        nodes_dict=hetero_nodes,
        source_node_types=["hex"],
        target_node_types=["street_connector"],
        k=1,
    )
    hetero_edges.update(bridged_edges)
    hetero_edges[bridge_key]["travel_time_sec"] = bridged_edges[bridge_key].length / WALKING_SPEED_MPS
    
    walk_edges = None
    if do_metapaths or do_clustering:
        metapath_key = ("hex", "15_min_walk", "hex")
        hetero_nodes, hetero_edges_new = c2g.add_metapaths_by_weight(
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
        walk_edges = hetero_edges_new[metapath_key]
        
    labels = None
    if do_clustering:
        print("Fetching Overture and running clustering...")
        overture_data = c2g.data.load_overture_data(area=boundary_polygon, types=["place", "land_use"])
        pois_gdf = overture_data["place"].to_crs(epsg=epsg)
        pois_gdf["categories"] = pois_gdf.get("categories", None)
        pois_gdf["functional_class"] = pois_gdf["categories"].apply(classify_poi)
        save_unmapped_categories()
        pois_gdf["geometry"] = pois_gdf.geometry.centroid
        hex_poly_gdf = hex_gdf.set_geometry("hex_polygon")
        joined_pois = gpd.sjoin(pois_gdf, hex_poly_gdf, how="inner", predicate="within")
        poi_counts = joined_pois.groupby("h3_index").size().rename("poi_count")
        hex_gdf["poi_count"] = hex_gdf.index.map(poi_counts).fillna(0).astype(int)

        landuse_gdf = overture_data["land_use"].to_crs(epsg=epsg)
        if "subtype" in landuse_gdf.columns and "class" in landuse_gdf.columns:
            landuse_gdf["landuse_category"] = landuse_gdf["subtype"].fillna(landuse_gdf["class"])
        else:
            landuse_gdf["landuse_category"] = "unknown"
            
        landuse_ratios, dominant_landuse = compute_land_use_ratios(hex_gdf, landuse_gdf)
        save_unmapped_land_use_categories()
        hex_gdf["landuse"] = dominant_landuse
        if not landuse_ratios.empty:
            hex_gdf[landuse_ratios.columns] = landuse_ratios

        scaler = StandardScaler()
        X_counts = scaler.fit_transform(np.log1p(hex_gdf[["poi_count"]]).to_numpy(dtype=float))
        hex_gdf[["poi_count"]] = X_counts

        landuse_feature_cols = [
            column for column in hex_gdf.columns
            if isinstance(column, str) and column.startswith("land_use_") and column.endswith("_ratio")
        ]
        if landuse_feature_cols:
            landuse_scaler = StandardScaler()
            hex_gdf[landuse_feature_cols] = landuse_scaler.fit_transform(
                hex_gdf[landuse_feature_cols].astype(float).to_numpy()
            )
        
        feature_cols = ["poi_count"] + landuse_feature_cols

        print("Converting to PyG and running fast GAE...")
        homo_data = c2g.gdf_to_pyg(
            nodes=hex_gdf,
            edges=walk_edges,
            node_feature_cols=feature_cols,
            edge_feature_cols=["travel_time_sec"],
        )
        
        class GATEncoder(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
                super().__init__()
                self.dropout = dropout
                self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
                self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)

            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                return self.conv2(x, edge_index, edge_weight)

        in_channels = homo_data.num_node_features
        model = GAE(GATEncoder(in_channels, 16, 8))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = homo_data.clone().to(device)
        edge_weight = data.edge_attr.view(-1).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(50): # few epochs for thumbnail
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index, edge_weight)
            loss = model.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index, edge_weight)
        
        Z_norm = normalize(z.cpu().numpy(), norm='l2')
        kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(Z_norm)
        hex_gdf["cluster"] = labels

    return hex_gdf, street_nodes, street_edges, bridged_edges[bridge_key], walk_edges, labels

def create_thumbnail():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(28, 9), dpi=300, facecolor='#0f0f0f')

    # Font dictionary for stunning titles
    title_font = {'fontsize': 40, 'fontweight': 'bold', 'color': 'white', 'family': 'sans-serif'}
    # Run processing ONCE to save time and ensure consistent data
    hex_gdf, street_nodes, street_edges, bridges, walk_edges, labels = process_city(
        AREA_NAME, EPSG, h3_res=H3_RESOLUTION, mode="all", do_metapaths=True, do_clustering=True
    )

    # Get consistent bounds
    bounds = hex_gdf.total_bounds # [minx, miny, maxx, maxy]
    margin_x = (bounds[2] - bounds[0]) * 0.05
    margin_y = (bounds[3] - bounds[1]) * 0.05
    xlim = (bounds[0] - margin_x, bounds[2] + margin_x)
    ylim = (bounds[1] - margin_y, bounds[3] + margin_y)

    # ----------------------------------------------------- #
    # Panel 1: STREETS AND BRIDGES
    # ----------------------------------------------------- #
    ax = axes[0]
    ax.set_facecolor('#0f0f0f')
    
    # Plot streets with a beautiful neon glow effect (pink/magenta)
    street_edges.plot(ax=ax, color='#ff00aa', linewidth=4, alpha=0.1)
    street_edges.plot(ax=ax, color='#ff00aa', linewidth=2, alpha=0.3)
    street_edges.plot(ax=ax, color='#ffffff', linewidth=0.6, alpha=0.9)
    
    # Plot hex polygons (boundaries only)
    hex_poly_gdf = hex_gdf.set_geometry("hex_polygon")
    hex_poly_gdf.boundary.plot(ax=ax, color='#555555', linewidth=0.8, alpha=0.7)
    
    # Plot bridges brightly (cyan/teal for contrast) with neon glow
    bridges.plot(ax=ax, color='#00e5ff', linewidth=4, alpha=0.2)
    bridges.plot(ax=ax, color='#00e5ff', linewidth=1.5, alpha=0.9)
    
    # Plot hex centroids with a glow
    hex_gdf.plot(ax=ax, color='#00e5ff', markersize=30, alpha=0.3, zorder=4)
    hex_gdf.plot(ax=ax, color='#ffffff', markersize=6, zorder=5)

    ax.set_title("1. Streets Network", fontdict=title_font, pad=35)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.text(0.5, -0.05, "© OpenStreetMap contributors", transform=ax.transAxes, 
            ha='center', va='top', color='#cccccc', fontsize=14, fontweight="bold")

    # ----------------------------------------------------- #
    # Panel 2: 15-MIN WALK METAPATHS
    # ----------------------------------------------------- #
    ax = axes[1]
    ax.set_facecolor('#0f0f0f')
    
    # Plot hex polygons (boundaries only)
    hex_poly_gdf.plot(ax=ax, facecolor='#161616', edgecolor='#333333', linewidth=0.6)
    
    import matplotlib.cm as cm
    vmin, vmax = 0, 900
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdYlBu

    # Plot Walk Edges (Metapaths) colored by travel time
    walk_edges.plot(
        ax=ax, 
        column='travel_time_sec', 
        cmap=cmap, 
        norm=norm, 
        linewidth=1.0, 
        alpha=0.9
    )
    
    # Plot hex centroids with a beautiful glow
    hex_gdf.plot(ax=ax, color='#00e5ff', markersize=45, alpha=0.2, zorder=4)
    hex_gdf.plot(ax=ax, color='#00e5ff', markersize=25, alpha=0.5, zorder=5)
    hex_gdf.plot(ax=ax, color='#ffffff', markersize=10, alpha=1.0, zorder=6)

    ax.set_title("2. 15-Minute Walkability", fontdict=title_font, pad=35)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')

    # ----------------------------------------------------- #
    # Panel 3: GEOAI CLUSTERING
    # ----------------------------------------------------- #
    ax = axes[2]
    ax.set_facecolor('#0f0f0f')
    
    unique_labels = sorted(hex_gdf["cluster"].unique())
    n_clusters = len(unique_labels)
    # Use a vibrant colormap for clusters (e.g., turbo or vivid plasma)
    cmap = sns.color_palette("turbo", n_clusters)
    colors = [mcolors.to_hex(cmap[label]) for label in hex_gdf["cluster"]]

    c2g.plot_graph(
        nodes=hex_poly_gdf,
        ax=ax,
        node_color=colors,
        node_edgecolor='#ffffff',
        node_alpha=0.85,
        linewidth=0.2, # thin white boundaries for clusters look sleek
        bgcolor='#0f0f0f'
    )
    ax.set_facecolor('#0f0f0f')

    ax.set_title("3. Clustering with GNNs", fontdict=title_font, pad=35)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    
    # Use tight_layout to make it all fit beautifully, but rect gives space for suptitle
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.82, wspace=0.1)
    
    output_path = os.path.join(IMG_DIR, "workshop_thumbnail.jpg")
    plt.savefig(output_path, dpi=300, facecolor='#0f0f0f', bbox_inches='tight', pad_inches=0.3)
    print(f"\nSaved beautiful thumbnail to {output_path}")

    # Combine the logo with the generated thumbnail in a square layout
    try:
        from PIL import Image
        bg = Image.open(output_path)
        logo_path = os.path.join(IMG_DIR, "city2graph_logo_main_dark.png")
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            
            # Improve layout by calculating padding relative to the image size
            # Add extra buffer specifically for the logo for a clean separation
            padding_x = int(bg.width * 0.05)
            padding_y = int(bg.height * 0.15)
            logo_buffer_y = int(bg.height * 0.20)
            
            new_width = bg.width + padding_x * 2
            
            logo_aspect = logo.width / logo.height
            
            # Make the logo take up a balanced portion of the width
            target_logo_width = int(new_width * 0.65)
            target_logo_height = int(target_logo_width / logo_aspect)
            
            logo = logo.resize((target_logo_width, target_logo_height), Image.Resampling.LANCZOS)
            
            # Calculate height to perfectly wrap the layout
            content_height = target_logo_height + logo_buffer_y + bg.height
            base_height = padding_y + content_height + padding_y
            
            # Ensure the total aspect of the output figure is square
            size = max(new_width, base_height)
            new_width = size
            new_height = size
                
            canvas = Image.new('RGB', (new_width, new_height), '#0f0f0f')
            
            # Center everything vertically and horizontally
            start_y = (new_height - content_height) // 2
            
            logo_x = (new_width - target_logo_width) // 2
            logo_y = start_y
            
            bg_x = (new_width - bg.width) // 2
            bg_y = start_y + target_logo_height + logo_buffer_y
            
            canvas.paste(bg, (bg_x, bg_y))
            
            if logo.mode == 'RGBA':
                canvas.paste(logo, (logo_x, logo_y), logo)
            else:
                canvas.paste(logo, (logo_x, logo_y))
                
            # If the image is very large, optionally scale it down so it hasn't got too many pixels
            MAX_SIZE = 4096
            print(canvas.width)
            if canvas.width > MAX_SIZE:
                canvas = canvas.resize((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
                
            canvas.save(output_path, quality=95)
            print("Successfully tightened the layout with a neat aspect ratio.")
    except ImportError:
        print("Note: Install 'Pillow' (e.g. pip install Pillow) to automatically add the logo.")



if __name__ == "__main__":
    create_thumbnail()
