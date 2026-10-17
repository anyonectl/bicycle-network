import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import LineString

# TODO: bias terms for each node in the VNS?
# TODO: needs some way to stop if no node in VNS is possible. boolean in irs dict?
def next_node(path_id: int, previous_node: str, irs: dict, seed_points: gpd.GeoDataFrame, rng: np.random.Generator):
    # Calculate probabilities for each node in the VNS
    filter = seed_points.loc[
        seed_points["id"] == previous_node, ["vns"]
    ].values.tolist()[0][0]
    vns = seed_points.loc[seed_points["id"].isin(filter)].sort_values(
        by=["total_demand"], ascending=False
    )
    activity_sum = float(vns["total_demand"].sum())
    vns["probability"] = vns["total_demand"] / activity_sum

    # Select next node and add to dict
    tries = vns.shape[0]
    next_node = ""
    for _ in range(tries):
        next_node = rng.choice(
            a=vns["id"], p=normalise_probabilities(vns["probability"])
        )
        if next_node not in irs[path_id]["nodes"]:
            irs[path_id]["nodes"].append(next_node)
            break
        else:
            vns = vns[(vns["id"] != next_node)]
    finished = vns.shape[0] == 0  # stop if no node in VNS is possible
    return (next_node, irs, finished)


def generate_route(k: int, max_paths: int, max_nodes: int, seed_points: gpd.GeoDataFrame, rng: np.random.Generator):
    """
    Form Initial Node Set (INS) from the first k nodes. k must be larger than max_paths
    """
    irs = {}
    ins = (
        seed_points.copy()
        .sort_values("total_demand", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    # Select first nodes
    for i in range(max_paths):
        # Calculate probabilities of selection a node j as first node
        activity_sum = float(ins["total_demand"].sum())
        ins["probability"] = ins["total_demand"] / activity_sum

        # Select first node
        first_node = rng.choice(
            a=ins["id"], p=normalise_probabilities(ins["probability"])
        )

        # Add to dict
        irs[i] = {"nodes": [first_node]}

        # Remove selected node from INS
        ins = ins[(ins["id"] != first_node)]

        # Build paths
        previous_node = first_node
        for _ in range(max_nodes - 1):  # -1 because we already added first_node
            tmp_next, tmp_irs, finished = next_node(i, previous_node, irs, seed_points, rng)
            if finished:  # stop if no node in VNS is possible
                break
            previous_node = tmp_next
            irs = tmp_irs

    return irs

def generate_population(n: int, k: int, max_paths: int, max_nodes: int, seed_points: gpd.GeoDataFrame, rng: np.random.Generator) -> gpd.GeoDataFrame:
    """
    Create population of n networks (route sets).
    Each network will contain the starting nodes selected from the k nodes with highest demand.
    """
    pop_list = []
    for i in range(n):
        gdf = create_gdf(generate_route(k, max_paths, max_nodes, seed_points, rng), seed_points)
        gdf["network_id"] = i
        # TODO: save_plot(gdf)
        pop_list.append(gdf)

    pop_gdf = pd.concat(pop_list)
    # TODO: save as geojson here?
    return pop_gdf

def normalise_probabilities(df: pd.Series) -> np.ndarray:
    p = df.to_numpy(dtype=np.float64)
    return p / np.sum(p, dtype=np.float64)

def create_gdf(irs: dict, seed_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Create dict for network
    d = {
        "path_id": [path_id for path_id in irs],
        "start_id": [path.get("nodes")[:][0] for _, path in irs.items()],
        "end_id": [path.get("nodes")[:][-1] for _, path in irs.items()],
        "path": [path.get("nodes") for _, path in irs.items()],
        # 'demand_covered': [],
        # 'geometry': []
    }

    # Create geometries
    id_to_centroid = seed_points.set_index("id")["centroid"].to_dict()
    geom = [
        LineString([id_to_centroid[point_id] for point_id in path])
        for path in d["path"]
    ]

    # Create dataframe
    network_gdf = gpd.GeoDataFrame(d, geometry=geom, crs=None)

    return network_gdf