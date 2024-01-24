import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import warnings


class Ploter:
    def __init__(self, DataProcessor):
        self.DP = DataProcessor

    def plot_edges(self, edges_weights, edges_ids):
        blocks = len(list(edges_weights.values())[0])

        for t in range(blocks):
            plot_nodes_indices = np.ones(len(self.DP.nodes_coo), dtype=bool)
            for p_idx in self.DP.producers_indices:
                plot_nodes_indices[int(p_idx)] = False
            for p_idx in self.DP.storage_indices:
                plot_nodes_indices[int(p_idx)] = False

            plt.scatter(
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 0],
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 1],
                c="b",
                s=20,
                label="consumer",
                alpha=0.5,
                linewidths=0,
            )

            plt.scatter(
                np.array(self.DP.nodes_coo)[np.array(self.DP.storage_indices), 0],
                np.array(self.DP.nodes_coo)[np.array(self.DP.storage_indices), 1],
                c="r",
                s=20,
                label="producer",
                alpha=0.5,
                linewidths=0,
            )
            # plt.scatter(splits_coo[:, 0], splits_coo[:, 1], c="g", s=10, label="split")
            # plt.scatter([125514.7, 125524.2], [ 481560.5,  481704.], c='g', s=16, label = "split")
            # plt.scatter(empty_node_coo[:, 0], empty_node_coo[:, 1], c="r", s=8, label="split")
            for e_idx, (edge_coordinates, edge_dia) in enumerate(
                zip(self.DP.edges_coordinates, self.DP.edges_dia)
            ):
                ec = np.array(edge_coordinates)
                e_id = edges_ids.get(e_idx)
                if e_id is None:
                    warnings.warn(
                        "edge idx %s cannot be find in the simulator." % e_idx
                        + " This should only happen with edges between storage and producer when storage is used as producer"
                    )
                    continue
                e_w = edges_weights[e_id][t]
                color = "black"
                if e_w == -1:
                    e_w = 1
                    color = "red"
                # plt.plot(ec[:,0], ec[:,1], c='black', alpha=0.5, linewidth=edge_dia/200)
                plt.plot(ec[:, 0], ec[:, 1], c=color, alpha=1, linewidth=e_w)

            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.show()
            # break

    def plot_nodes(self, consumers_weights, nodes_ids, edges_ids):
        blocks = len(list(consumers_weights.values())[0])
        min_weight = min([min(v) for v in consumers_weights.values()])
        max_weight = max([max(v) for v in consumers_weights.values()])
        for t in range(blocks):
            plot_nodes_indices = np.ones(len(self.DP.nodes_coo), dtype=bool)
            for p_idx in self.DP.producers_indices:
                plot_nodes_indices[int(p_idx)] = False
            for p_idx in self.DP.storage_indices:
                plot_nodes_indices[int(p_idx)] = False

            nValues = [v[t] for v in consumers_weights.values()]

            # setup the normalization and the colormap
            normalize = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
            colormap = cm.jet

            fig = plt.figure(figsize=(9, 6))
            plt.scatter(
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 0],
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 1],
                c=colormap(normalize(nValues)),
                s=20,
                # label="consumer",
                alpha=1,
                linewidths=0,
            )

            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(nValues)
            plt.colorbar(scalarmappaple)

            for e_idx, (edge_coordinates, edge_dia) in enumerate(
                zip(self.DP.edges_coordinates, self.DP.edges_dia)
            ):
                ec = np.array(edge_coordinates)
                e_id = edges_ids.get(e_idx)
                # if e_id is None:
                #     if t == 0:
                #         warnings.warn(
                #             "edge idx %s cannot be find in the simulator." % e_idx
                #             + " This should only happen with edges between storage and producer when storage is used as producer"
                #         )
                # continue
                plt.plot(ec[:, 0], ec[:, 1], c="black", alpha=0.5, linewidth=0.5)

            plt.xticks([])
            plt.yticks([])
            plt.show()
            # fig.savefig(r"C:\Users\84186\Documents\WTG\flex-heat\vattenfall\plots\inlet_temp_random_Demand\{:02d}.png".format(t), dpi=fig.dpi)
            fig.clear()

    def plot_nodes_edges(
        self,
        consumers_weights,
        nodes_ids,
        edges_weights,
        edges_ids,
        demands,
    ):
        blocks = len(list(consumers_weights.values())[0])
        min_weight = min([min(v) for v in consumers_weights.values()])
        max_weight = max([max(v) for v in consumers_weights.values()])
        for t in range(blocks):
            plot_nodes_indices = np.ones(len(self.DP.nodes_coo), dtype=bool)
            for p_idx in self.DP.producers_indices:
                plot_nodes_indices[int(p_idx)] = False

            nValues = [v[t] for v in consumers_weights.values()]

            # setup the normalization and the colormap
            normalize = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
            colormap = cm.jet

            fig = plt.figure(figsize=(9, 6))
            plt.scatter(
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 0],
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 1],
                c=colormap(normalize(nValues)),
                s=20,
                # label="consumer",
                alpha=1,
                linewidths=0,
            )

            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(nValues)
            plt.colorbar(scalarmappaple)

            for e_idx, (edge_coordinates, edge_dia) in enumerate(
                zip(self.DP.edges_coordinates, self.DP.edges_dia)
            ):
                ec = np.array(edge_coordinates)
                e_id = edges_ids.get(e_idx)
                if e_id is None:
                    if t == 0:
                        warnings.warn(
                            "edge idx %s cannot be find in the simulator." % e_idx
                            + " This should only happen with edges between storage and producer when storage is used as producer"
                        )
                    continue
                e_w = edges_weights[e_id][t]
                color = "black"
                if e_w == -1:
                    e_w = 3
                    color = "red"
                    print("violation at %s" % t)
                # plt.plot(ec[:,0], ec[:,1], c='black', alpha=0.5, linewidth=edge_dia/200)
                plt.plot(ec[:, 0], ec[:, 1], c=color, alpha=1, linewidth=e_w)

            plt.xticks([])
            plt.yticks([])
            # plt.legend()
            plt.title("total demand: {:.1f}.png".format(demands[t]) + " at time %s" % t)
            plt.show()

    def plot_edges_color(
        self,
        edges_weights,
        edges_ids,
        demands,
        sup_temps,
        fig_idx,
    ):
        blocks = len(list(edges_weights.values())[0])
        min_weight = min([min(v) for v in edges_weights.values()])
        max_weight = max([max(v) for v in edges_weights.values()])
        for t in range(blocks):
            plot_nodes_indices = np.ones(len(self.DP.nodes_coo), dtype=bool)
            for p_idx in self.DP.producers_indices:
                plot_nodes_indices[int(p_idx)] = False

            # setup the normalization and the colormap
            normalize = mcolors.Normalize(vmin=0, vmax=1)
            colormap = cm.jet
            nValues = [v[t] for v in edges_weights.values()]
            fig = plt.figure(figsize=(9, 6))
            plt.scatter(
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 0],
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 1],
                c="b",
                s=20,
                # label="consumer",
                alpha=1,
                linewidths=0,
            )
            violation = 0
            high_cap = 0
            for e_idx, (edge_coordinates, edge_dia) in enumerate(
                zip(self.DP.edges_coordinates, self.DP.edges_dia)
            ):
                ec = np.array(edge_coordinates)
                e_id = edges_ids.get(e_idx)
                if e_id is None:
                    if t == 0:
                        warnings.warn(
                            "edge idx %s cannot be find in the simulator." % e_idx
                            + " This should only happen with edges between storage and producer when storage is used as producer"
                        )
                    continue

                # plt.plot(ec[:,0], ec[:,1], c='black', alpha=0.5, linewidth=edge_dia/200)
                violation += edges_weights[e_id][t] >= 1
                high_cap += edges_weights[e_id][t] >= 0.8
                plt.plot(
                    ec[:, 0],
                    ec[:, 1],
                    c=colormap(normalize(edges_weights[e_id][t])),
                    alpha=1,
                    linewidth=1,
                )

            plt.xticks([])
            plt.yticks([])
            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(nValues)
            plt.colorbar(scalarmappaple)
            # plt.legend()
            plt.title(
                "total demand: {:.1f}".format(demands[t])
                + ", at sup_temp: {:.0f}".format(sup_temps[t])
                + "\n pipe violation count: %s" % violation
                + ", large capacity count (>80%%): %s" % high_cap
            )
            plt.show()
            # fig.savefig(r"C:\Users\84186\Documents\WTG\flex-heat\vattenfall\plots\temp_demand_profile\{:02d}.png".format(fig_idx), dpi=fig.dpi)

    def plot_nodes_clustered(self, consumers_weights, clustered_connection):
        blocks = len(list(consumers_weights.values())[0])
        min_weight = min([min(v) for v in consumers_weights.values()])
        max_weight = max([max(v) for v in consumers_weights.values()])
        for t in range(blocks):
            plot_nodes_indices = np.ones(len(self.DP.nodes_coo), dtype=bool)
            for p_idx in self.DP.producers_indices:
                plot_nodes_indices[int(p_idx)] = False
            for p_idx in self.DP.storage_indices:
                plot_nodes_indices[int(p_idx)] = False

            nValues = [v[t] for v in consumers_weights.values()]

            # setup the normalization and the colormap
            normalize = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
            colormap = cm.jet

            fig = plt.figure(figsize=(9, 6))
            plt.scatter(
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 0],
                np.array(self.DP.nodes_coo)[plot_nodes_indices, 1],
                c=colormap(normalize(nValues)),
                s=20,
                # label="consumer",
                alpha=1,
                linewidths=0,
            )

            cluster_node_branch_indices = [[c[0], c[1]] for c in clustered_connection]
            cluster_node_branch_indices = np.unique(
                np.array(cluster_node_branch_indices).flatten()
            )
            cluster_node_indices = cluster_node_branch_indices[
                (
                    (cluster_node_branch_indices < len(self.DP.nodes_coo))
                    & (cluster_node_branch_indices > 0)
                )
            ]
            cluster_branch_indices = cluster_node_branch_indices[
                cluster_node_branch_indices >= len(self.DP.nodes_coo)
            ]
            plt.scatter(
                np.array(self.DP.nodes_coo)[cluster_node_indices, 0],
                np.array(self.DP.nodes_coo)[cluster_node_indices, 1],
                c="white",
                s=30,
                # label="consumer",
                alpha=1,
                linewidths=3,
                edgecolors="black",
            )
            plt.scatter(
                np.array(self.DP.splits_coo)[
                    cluster_branch_indices - len(self.DP.nodes_coo), 0
                ],
                np.array(self.DP.splits_coo)[
                    cluster_branch_indices - len(self.DP.nodes_coo), 1
                ],
                c="white",
                s=10,
                # label="consumer",
                alpha=1,
                linewidths=2,
                edgecolors="black",
            )

            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(nValues)
            plt.colorbar(scalarmappaple)

            for e_idx, (edge_coordinates, edge_dia) in enumerate(
                zip(self.DP.edges_coordinates, self.DP.edges_dia)
            ):
                ec = np.array(edge_coordinates)

                plt.plot(ec[:, 0], ec[:, 1], c="black", alpha=0.5, linewidth=0.5)

            plt.xticks([])
            plt.yticks([])
            plt.show()
            # fig.savefig(r"C:\Users\84186\Documents\WTG\flex-heat\vattenfall\plots\inlet_temp_random_Demand\{:02d}.png".format(t), dpi=fig.dpi)
            fig.clear()
