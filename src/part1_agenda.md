# Part 1 Agenda: Graph Data Engineering, Spatial Network Analysis, and GNNs

## Context & Design Rationale

Part 1 is the **conceptual and foundational** half of the workshop. It should give participants the "why" and "how" behind every tool and technique they will use hands-on in Part 2. By the end of Part 1, attendees should be able to:

- Explain what a spatial graph is and why hexagonal tessellations are useful
- Understand how geospatial data sources (OSM, Overture Maps, GTFS) map onto graph structures
- Navigate key GNN architectures at a high level (message passing, GCN, GAT, autoencoders)
- Understand the PyTorch Geometric data model (`Data`, `HeteroData`)

---

## Proposed Agenda

### 0. Introduction & Workshop Logistics (~5 min)
- Who the workshop is for (recap target audience from README)
- Overall workshop structure: Part 1 (concepts + demo) → Part 2 (hands-on)
- Environment check: `uv sync` should already be done; quick sanity check

---

### 1. From Geospatial Data to Graphs (~15 min)

#### 1.1 Why graphs for geospatial data?
- Limitations of raster/tabular representations for modelling **relationships**
- Graphs capture topology, proximity, and connectivity naturally
- Real-world motivations: walkability analysis, service area modelling, urban typology classification

#### 1.2 Spatial tessellation with H3 hexagons
- Why hexagons? (uniform adjacency, equal area, hierarchical resolution)
- H3 resolution trade-offs (res 9 ≈ 0.1 km², res 10 ≈ 0.015 km²)
- Live demo: generate H3 cells for a small area, visualise with GeoPandas

#### 1.3 Enriching nodes with spatial context
- **Overture Maps**: POIs (places) and land-use polygons
- POI classification pipeline (functional categories → per-hex feature vector)
- Land-use ratio computation (spatial overlay → occupancy proportions)
- Brief mention of other data sources: GTFS (transit), census, sensor data

---

### 2. Building Spatial Networks (~15 min)

#### 2.1 Street networks with OSMnx
- Fetching walkable graphs from OpenStreetMap
- NetworkX ↔ GeoDataFrame conversions (`c2g.nx_to_gdf`)
- Edge attributes: length, travel time

#### 2.2 Contiguity graphs (H3 neighbours)
- Queen contiguity: adjacent hexagons share an edge or vertex
- `c2g.contiguity_graph()` — fast alternative to street routing
- When to use contiguity vs. street routing

#### 2.3 Heterogeneous graphs and metapaths
- **Heterogeneous graph**: multiple node types (hex, street connector) and edge types
- **Bridge edges**: connecting hex centroids to nearest street intersections
- **Metapaths**: "15-minute walk" reachability via Dijkstra on the street network
- Concept: collapsing a multi-layer network into a single-layer hex–hex graph

#### 2.4 Hands-on check / live demo
- Show a small spatial graph (nodes, edges, features) in matplotlib
- Explore the interactive folium map from a pre-computed example

---

### 3. Introduction to Graph Neural Networks (~20 min)

#### 3.1 Why GNNs? (vs. CNNs, MLPs on tabular data)
- Irregular, non-Euclidean structure of spatial graphs
- Message passing: each node aggregates information from its neighbours
- Analogy: a hexagon "learns" about its surrounding urban context

#### 3.2 Key GNN architectures (high-level)
- **GCN** (Graph Convolutional Network): spectral convolution, simple mean aggregation
- **GAT** (Graph Attention Network): learnable attention weights → different neighbours contribute differently
- Why we use GAT in this workshop (edge-weighted attention suits travel-time graphs)

#### 3.3 Graph Autoencoders (GAE) for unsupervised learning
- Encoder–decoder paradigm
- Encoder: GATConv layers → low-dimensional node embeddings
- Decoder: inner-product reconstruction of the adjacency matrix
- Loss function: binary cross-entropy on reconstructed edges
- What the learned embeddings represent: compressed urban "fingerprints"

#### 3.4 From spatial data to tensors — PyTorch Geometric
- `Data` object: `x` (node features), `edge_index`, `edge_attr`
- City2Graph bridge: `c2g.gdf_to_pyg()` and `c2g.pyg_to_gdf()`
- Feature preprocessing recap: log-transform, StandardScaler, MinMaxScaler

---

### 4. Putting It All Together — End-to-End Pipeline Overview (~10 min)

Walk through the full pipeline that participants will implement in Part 2:

```
Area selection → H3 tessellation → Overture data fetch
     → POI / land-use feature engineering
     → Graph construction (contiguity or street-based)
     → Feature scaling → PyG conversion
     → GAE training (500 epochs)
     → Embedding extraction → Clustering (HDBSCAN / K-Means)
     → Interactive map export
```

- Show the pre-computed clustering result image ([img/clustering_result.png](file:///Users/yutasato/Projects/Liverpool/city2graph-workshop/img/clustering_result.png))
- Discuss interpretation: what do the clusters mean urbanistically?
- Preview: in Part 2 you will run this for **your own city**

---

### 5. Q&A / Break before Part 2 (~5 min)
- Open questions
- Transition to Part 2 hands-on notebook

---

## Timing Summary

| Section | Topic | Duration |
|---------|-------|----------|
| 0 | Introduction & Logistics | ~5 min |
| 1 | From Geospatial Data to Graphs | ~15 min |
| 2 | Building Spatial Networks | ~15 min |
| 3 | Introduction to GNNs | ~20 min |
| 4 | End-to-End Pipeline Overview | ~10 min |
| 5 | Q&A / Break | ~5 min |
| | **Total** | **~70 min** |

## Key Decisions to Consider

1. **Slide deck vs. notebook?** Part 1 is described as "Notebook WIP" in the README. A Jupyter notebook with markdown + code demos would be consistent, but a slide deck may work better for the conceptual GNN sections (diagrams of message passing, GAT attention, autoencoder architecture). A hybrid approach (slides for Section 3, notebook for Sections 1-2 and 4) is also viable.

2. **Depth of GNN theory**: The target audience has basic ML knowledge but no GNN experience. The current plan keeps GNN theory at an intuitive level (message passing analogy, attention weights concept) without going into matrix algebra. Do you want to go deeper (e.g., spectral vs. spatial convolutions)?

3. **Live coding vs. pre-built demos**: For Sections 1–2, you could either live-code small examples or show pre-computed outputs. Live coding is more engaging but riskier with network-dependent APIs (Overture, OSMnx).

4. **Duration flexibility**: The current plan targets ~70 min. If the total workshop slot allows more time for Part 1, Sections 1 and 3 could be expanded. If less, Section 2 could be compressed by focusing only on contiguity graphs.
