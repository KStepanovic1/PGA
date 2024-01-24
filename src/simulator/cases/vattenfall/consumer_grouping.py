import numpy as np
import copy
from sklearn.cluster import AgglomerativeClustering

from .data_processing_ams import DataProcessor as DP

# build connection from each consumer to the producer,
# through all splits. distance is also included
def build_connection():
    edges_through_empty_nodes_reverse = {
        v: k for k, v in DP.edges_connected_through_empty_nodes.items()
    }

    connection_lists = []
    distance_lists = []
    for consumer_n_idx, node_data in enumerate(DP.nodes_list):
        if (consumer_n_idx in DP.producers_indices) or (
            consumer_n_idx in DP.storage_indices
        ):
            continue

        terminate = False
        n_idx = consumer_n_idx
        edge_idx, _ = DP.node_edge_connection[n_idx]
        connection_list = [n_idx]
        distance_list = []
        while not terminate:

            if n_idx < len(DP.nodes_list):
                edge_idx, _ = DP.node_edge_connection[n_idx]
            else:
                edge_idx = None
                for e_idx, _ in DP.split_edge_connections[n_idx - len(DP.nodes_list)]:
                    if DP.edge_to_node_connection[e_idx][1] == n_idx:
                        edge_idx = e_idx

            distance = DP.edges_len[edge_idx]
            downstream_edge_idx = edge_idx
            upstream_edge_idx = edges_through_empty_nodes_reverse.get(
                downstream_edge_idx
            )
            while upstream_edge_idx is not None:
                distance += DP.edges_len[upstream_edge_idx]
                downstream_edge_idx = upstream_edge_idx
                upstream_edge_idx = edges_through_empty_nodes_reverse.get(
                    downstream_edge_idx
                )

            upstream_node_idx, _ = DP.edge_to_node_connection[downstream_edge_idx]
            connection_list.append(upstream_node_idx)
            distance_list.append(distance)
            if upstream_node_idx < len(DP.nodes_list):
                assert upstream_node_idx in DP.storage_indices
                terminate = True
                connection_lists.append(np.array(connection_list))
                distance_lists.append(np.array(distance_list))
            else:
                n_idx = upstream_node_idx

    # connection_lists: index of all passed nodes/splits from a consumer till producer
    # distance_lists: the distance between each two adjacent node in the connection_lists
    return connection_lists, distance_lists


# cluster all consumers
def graph_hierarchical_cluster(connection_lists, distance_lists, cluster_number):
    distance_matrix = np.zeros((len(connection_lists), len(connection_lists)))
    for i, path_i in enumerate(connection_lists):
        for j in range(i + 1, len(connection_lists)):
            path_j = connection_lists[j]
            min_len = min(len(path_i), len(path_j))
            split_idx = np.argmin(path_i[::-1][:min_len] == path_j[::-1][:min_len])
            distance = np.sum(distance_lists[i][::-1][split_idx:]) + np.sum(
                distance_lists[i][::-1][split_idx:]
            )
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    distance_threshold = np.max(
        distance_matrix[
            np.arange(len(connection_lists)), np.argsort(distance_matrix, axis=1)[:, 2]
        ]
    )
    distance_matrix_binary = distance_matrix <= distance_threshold

    # overlap_matrix = np.zeros((len(connection_lists), len(connection_lists)))
    # for i, path_i in enumerate(connection_lists):
    #     for j in range(i+1, len(connection_lists)):
    #         path_j = connection_lists[j]
    #         min_len = min(len(path_i), len(path_j))
    #         overlap_count = np.sum(np.equal(path_i[::-1][:min_len],path_j[::-1][:min_len]))
    #         overlap_matrix[i,j] = overlap_count

    # cluster_matrix = np.zeros((len(connection_lists), len(connection_lists)))
    # count = 1

    # for idx in np.argsort(overlap_matrix.flatten())[::-1]:
    #     row_idx = idx//len(connection_lists)
    #     col_idx = idx%len(connection_lists)

    #     def find_exist_connection(r_idx, c_idx, looked_indices=[]):
    #         looked_indices.append(r_idx)
    #         x = np.where(cluster_matrix[r_idx])[0]
    #         if c_idx in x:
    #             return True
    #         elif len(x)>0:
    #             found = False
    #             for x_ in x:
    #                 if x_ not in looked_indices:
    #                     found = found|find_exist_connection(x_, c_idx, looked_indices)
    #             return found
    #         else:
    #             return False

    #     if (
    #         not find_exist_connection(row_idx, col_idx)
    #         # (np.sum(cluster_matrix[row_idx]) == 0)
    #         # | (np.sum(cluster_matrix[col_idx]) == 0)
    #     ):
    #         cluster_matrix[row_idx, col_idx] = count
    #         cluster_matrix[col_idx, row_idx] = count
    #         count += 1

    #     if count == len(connection_lists):
    #         break
    # else:
    #     raise Exception

    # assert np.all(np.sum(cluster_matrix, axis=0) != 0)
    # assert np.all(np.sum(cluster_matrix, axis=1) != 0)

    # cluster_matrix = np.array(cluster_matrix, dtype=bool)
    indices = np.ones(len(DP.nodes_coo), dtype=bool)
    indices[DP.producers_indices] = 0
    indices[DP.storage_indices] = 0
    clustered = AgglomerativeClustering(
        n_clusters=cluster_number, linkage="ward", connectivity=distance_matrix_binary
    ).fit(DP.nodes_coo[indices])
    labels = clustered.labels_

    return labels

    # clustered = {}
    # for i in range(1,count):
    # rows_indices, _ = np.where(cluster_matrix == i)
    # row_idx, col_idx = rows_indices

    # def convert_consumer_idx_to_node_idx(consumer_idx):
    #     x = list(DP.producers_indices)
    #     x.extend(list(DP.storage_indices))
    #     x = np.sort(x)
    #     for x_ in x:
    #         if consumer_idx >= x_:
    #             consumer_idx += 1
    #     return consumer_idx

    # def flatten(x):
    #     result = []
    #     for el in x:
    #         if hasattr(el, "__iter__") and not isinstance(el, str):
    #             result.extend(flatten(el))
    #         else:
    #             result.append(el)
    #     return result

    # previous_clusters1 = np.where(
    #     (cluster_matrix[row_idx] < i)
    #     & (cluster_matrix[row_idx] != 0)
    # )[0]
    # previous_clusters2 = np.where(
    #     (cluster_matrix[col_idx] < i)
    #     & (cluster_matrix[col_idx] != 0)
    #     )[0]

    # consumer1_idx = convert_consumer_idx_to_node_idx(row_idx)
    # consumer2_idx = convert_consumer_idx_to_node_idx(col_idx)
    # if (len(previous_clusters1) == 0):
    #     if (len(previous_clusters2) == 0):
    #         if i != 1:
    #             old_cluster = copy.deepcopy(clustered[i-1])
    #             old_cluster.extend([consumer1_idx, consumer2_idx])
    #             clustered[i] = old_cluster
    #         else:
    #             clustered[i] = [consumer1_idx, consumer2_idx]
    #     else:
    #         old_cluster = copy.deepcopy(clustered[i-1])
    #         for j in range(len(old_cluster)):
    #             if consumer2_idx in flatten([old_cluster[j]]):
    #                 old_cluster[j] = [old_cluster[j], consumer1_idx]
    #                 clustered[i] = old_cluster
    #                 break
    #         else:
    #             raise Exception
    # elif len(previous_clusters2) == 0:
    #     if i == 114:
    #         print(col_idx, consumer2_idx,cluster_matrix[col_idx])
    #         exit()
    #     old_cluster = copy.deepcopy(clustered[i-1])
    #     for j in range(len(old_cluster)):
    #         if consumer1_idx in flatten([old_cluster[j]]):
    #             old_cluster[j] = [old_cluster[j], consumer2_idx]
    #             clustered[i] = old_cluster
    #             break
    #     else:
    #         raise Exception
    # else:

    #     old_cluster = copy.deepcopy(clustered[i-1])
    #     cluster_idx1, cluster_idx2 = None, None
    #     for j in range(len(old_cluster)):
    #         if consumer1_idx in flatten([old_cluster[j]]):
    #             cluster_idx1 = j

    #             assert consumer2_idx not in flatten([old_cluster[j]])
    #         elif consumer2_idx in flatten([old_cluster[j]]):
    #             cluster_idx2 = j
    #     assert (cluster_idx1 is not None) and (cluster_idx2 is not None)
    #     old_cluster[cluster_idx1] = [old_cluster[cluster_idx1], old_cluster[cluster_idx2]]
    #     old_cluster.pop(cluster_idx2)
    #     clustered[i] = old_cluster


# find center of all clusters and the splits that connect them
def build_clustered_network(connection_lists, distance_lists, labels):
    def find_common_path(paths_list):
        min_path_len = min([len(path) for path in paths_list])
        common_path_idx = np.ones(min_path_len, dtype=bool)
        for i in range(len(paths_list) - 1):
            common_path_idx = common_path_idx & np.equal(
                paths_list[i][::-1][:min_path_len],
                paths_list[i + 1][::-1][:min_path_len],
            )
        common_path = paths_list[0][::-1][:min_path_len][common_path_idx]
        return common_path

    connection_lists = np.array(connection_lists, dtype=object)
    distance_lists = np.array(distance_lists, dtype=object)
    dic_common_path = {}
    dic_last_common_split_idx = {}
    dic_furthest_distance = {}
    dic_edge_dia = {}
    dic_furthest_node_idx = {}

    for cluster_idx in range(np.max(labels) + 1):
        nodes_idx = np.where(labels == cluster_idx)[0]

        common_path = find_common_path(connection_lists[nodes_idx])
        last_common_split_idx = common_path[-1]

        # distance from the last common split to the furthest node
        furthest_distance = max(
            [
                np.sum(distance_list[: -len(common_path) + 1])
                for distance_list in distance_lists[nodes_idx]
            ]
        )
        furthest_node_idx = np.argmax(
            [
                np.sum(distance_list[: -len(common_path) + 1])
                for distance_list in distance_lists[nodes_idx]
            ]
        )
        furthest_node_idx = nodes_idx[furthest_node_idx]
        skip_indices = copy.copy(DP.producers_indices)
        skip_indices.extend(DP.storage_indices)
        skip_indices = np.sort(skip_indices)
        for i in skip_indices:
            if furthest_node_idx >= i:
                furthest_node_idx += 1

        dic_common_path[cluster_idx] = common_path
        dic_last_common_split_idx[cluster_idx] = last_common_split_idx
        dic_furthest_distance[cluster_idx] = furthest_distance
        dic_furthest_node_idx[cluster_idx] = furthest_node_idx

        edges_slots = DP.split_edge_connections[
            last_common_split_idx - len(DP.nodes_list)
        ]
        edge_dia = max([DP.edges_dia[edge] for edge, _ in edges_slots])
        dic_edge_dia[cluster_idx] = edge_dia

    consumer_idx_to_cluster_idx = {v: k for k, v in dic_furthest_node_idx.items()}
    clustered_connection = []
    for cluster_idx1 in range(np.max(labels) + 1):
        split_appeared_in_other_cluster = False
        last_shared_split_idx = -1
        for cluster_idx2 in range(np.max(labels) + 1):
            if cluster_idx1 == cluster_idx2:
                continue

            split_appeared_in_other_cluster = split_appeared_in_other_cluster | (
                dic_last_common_split_idx[cluster_idx1] in dic_common_path[cluster_idx2]
            )
            shared_split_idx = np.where(
                dic_common_path[cluster_idx1] == dic_last_common_split_idx[cluster_idx2]
            )[0]
            if len(shared_split_idx) > 0:
                if shared_split_idx[0] > last_shared_split_idx:
                    last_shared_split_idx = shared_split_idx[0]

        if not split_appeared_in_other_cluster:
            assert last_shared_split_idx >= 0
            nodes_idx = np.where(labels == cluster_idx1)[0]

            distance = max(
                [
                    np.sum(distance_list[:-last_shared_split_idx])
                    for distance_list in distance_lists[nodes_idx]
                ]
            )

            clustered_connection.append(
                [
                    dic_common_path[cluster_idx1][last_shared_split_idx],
                    dic_furthest_node_idx[cluster_idx1],
                    distance,
                    dic_edge_dia[cluster_idx1] / 1000,
                ]
            )

        else:
            clustered_connection.append(
                [
                    dic_last_common_split_idx[cluster_idx1],
                    dic_furthest_node_idx[cluster_idx1],
                    dic_furthest_distance[cluster_idx1],
                    dic_edge_dia[cluster_idx1] / 1000,
                ]
            )
            if last_shared_split_idx >= 0:
                nodes_idx = np.where(labels == cluster_idx1)[0]
                distance = np.sum(
                    distance_lists[nodes_idx[0]][
                        -len(dic_common_path[cluster_idx1]) + 1 : -last_shared_split_idx
                    ]
                )
                clustered_connection.append(
                    [
                        dic_common_path[cluster_idx1][last_shared_split_idx],
                        dic_last_common_split_idx[cluster_idx1],
                        distance,
                        dic_edge_dia[cluster_idx1] / 1000,
                    ]
                )
            else:

                nodes_idx = np.where(labels == cluster_idx1)[0]
                distance = np.sum(
                    distance_lists[nodes_idx[0]][
                        -len(dic_common_path[cluster_idx1]) + 1 :
                    ]
                )
                clustered_connection.append(
                    [
                        DP.storage_indices[0],
                        dic_last_common_split_idx[cluster_idx1],
                        distance,
                        dic_edge_dia[cluster_idx1] / 1000,
                    ]
                )

    # this is a list containing all connections, with the order of:
    # [upstream node idx, downstream node idx, distance, pipe diameter]
    return clustered_connection, consumer_idx_to_cluster_idx

    # for node1, node2, _, _ in connection_list:
    #     for node in [node1, node2]:
    #         if node == -1:
    #             print(node, DP.nodes_coo[DP.storage_indices[0]])
    #         elif node < len(DP.nodes_coo):
    #             print(node, DP.nodes_coo[dic_furthest_node_idx[node]])
    #         else:
    #             print(node, DP.splits_coo[node - len(DP.nodes_coo)])


if __name__ == "__main__":
    connection_lists, distance_lists = build_connection()
    labels = graph_hierarchical_cluster(connection_lists, distance_lists, 20)
    (clustered_connection, consumer_idx_to_cluster_idx) = build_clustered_network(
        connection_lists, distance_lists, labels
    )
    labels_value = {i: [labels[i]] for i in range(len(labels))}
    from .plot_grid import Ploter

    ploter = Ploter(DP)
    ploter.plot_nodes_clustered(labels_value, clustered_connection)
