"""
Please refer to the class 'DataProcessor' to learn how to use data from this script
"""
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer


class Geo_trans(object):
    """docstring for Geo_trans"""

    def __init__(self):
        crs = CRS.from_proj4(
            "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889"
            + " +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel"
            + " +towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725"
            + " +units=m +vunits=m +no_defs"
        )
        crs2 = CRS.from_epsg(4326)
        self.proj1 = Transformer.from_crs(crs, crs2)
        self.proj2 = Transformer.from_crs(crs2, crs)

    def from_meter(self, x, y):
        return self.proj1.transform(x, y)

    def to_meter(self, lon, la):
        return self.proj2.transform(lon, la)


class Pipe_params_loader:
    dia_vmax_chart = np.array(
        [
            [0, 0.5],
            [0.05, 0.9],
            [0.08, 1.2],
            [0.1, 1.3],
            [0.125, 1.5],
            [0.15, 1.75],
            [0.2, 2],
            [0.25, 2.2],
            [0.3, 2.6],
            [0.4, 2.8],
            [0.5, 3],
            [0.6, 3],
            [0.7, 3],
            [1, 3],
        ]
    )
    # thermal resistance k in K*m/W
    dia_k_chart = np.array(
        [
            [0, 10],
            [0.02, 6.21],
            [0.025, 5.07],
            [0.032, 4.92],
            [0.04, 4.26],
            [0.05, 3.8],
            [0.075, 3.07],
            [0.1, 2.93],
            [0.15, 2.09],
            [0.2, 1.9],
            [0.25, 1.97],
            [0.3, 1.69],
            [0.4, 1.64],
            [0.5, 1.69],
            [0.6, 1.36],
            [1, 1.36],
        ]
    )

    def get_vmax(self, dia):
        return np.interp(dia, self.dia_vmax_chart[:, 0], self.dia_vmax_chart[:, 1])

    def get_k(self, dia):
        return np.interp(dia, self.dia_k_chart[:, 0], self.dia_k_chart[:, 1])


def load_edge_vt():
    path = Path(__file__).parents[4] / "data/vattenfall/edges.csv"

    rows = []
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)
    rows = rows[1:]

    return rows


def convert_to_point(coo_string):
    coo_string = str(coo_string)
    start = coo_string.find("(") + 1
    end = coo_string.find(")")
    coo_string = coo_string[start:end].split()
    coo = np.array(coo_string, dtype=float)
    # coo = coo.round(-1)
    return coo


def load_node():
    path = Path(__file__).parents[4] / "data/vattenfall/nodes.csv"

    rows = []
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if row[1] == "KOUDE PLAN":
                continue
            rows.append(row)
    rows = rows[1:]

    return rows


def convert_to_lines(coo_string):
    coo_string = str(coo_string)
    start = coo_string.find("(") + 1
    end = coo_string.find(")")
    coo_string = coo_string[start:end].split(",")
    coo_string = [s.split() for s in coo_string]
    coo = np.array(coo_string, dtype=float)
    # coo = coo.round(-1)
    return coo


def load_edge():
    path = Path(__file__).parents[4] / "data/vattenfall/STADSWARMTEKOUDE_LIJN.csv"

    rows = []
    with open(path, "r") as file:
        csvreader = csv.reader(file, delimiter=";")
        for row in csvreader:
            if row[1] == "WARMTE TRANSPORT":
                rows.append(row)

    return rows


def trim_edge():
    rows = load_edge()
    geo_trans = Geo_trans()
    coordinates = [convert_to_lines(r[-4]) for r in rows]
    coordinates = [
        [geo_trans.to_meter(c[0], c[1]) for c in coos] for coos in coordinates
    ]

    edges_vt = load_edge_vt()
    coordinates_vt = [convert_to_lines(r[-1]) for r in edges_vt]
    pipe_dia_vt = [e[5] for e in edges_vt]
    xs = np.array([[c[0] for c in coo] for coo in coordinates_vt]).flatten()
    ys = np.array([[c[1] for c in coo] for coo in coordinates_vt]).flatten()
    x_range = [max(xs), min(xs)]
    y_range = [max(ys), min(ys)]

    def find_best_match(coo, coo_list):
        d_min = np.inf
        idx = None
        for i, coos in enumerate(coo_list):
            for coo2 in coos:
                d = np.abs(coo2[0] - coo[0]) + np.abs(coo2[1] - coo[1])
                if d < d_min:
                    d_min = d
                    idx = i
                    if d_min < 0.1:
                        return d_min, idx

        return d_min, idx

    list_keep = []
    pipe_dia_ams = []
    for i, edge in enumerate(coordinates):
        print(i)
        d_max = 0
        d_sum = 0
        count = 0
        if (
            (edge[0][0] > x_range[0] + 200)
            | (edge[0][0] < x_range[1] - 200)
            | (edge[0][1] > y_range[0] + 200)
            | (edge[0][1] < y_range[1] - 200)
        ):
            continue
        pipe_dias = []
        for pt in edge:
            distance, idx = find_best_match(pt, coordinates_vt)
            d_max = max(distance, d_max)
            d_sum += distance
            count += 1
            if d_max > 100:
                break
            pipe_dias.append(pipe_dia_vt[idx])
        print(d_max)
        if d_sum / count < 10:
            list_keep.append(i)
            unique_dia, count = np.unique(pipe_dias, return_counts=True)
            dia = unique_dia[np.argmax(count)]
            dia = dia.split("/")
            dia[0] = dia[0][2:]
            pipe_dia_ams.append(np.array(dia, dtype=int))
    print(len(list_keep), pipe_dia_ams)

    np.save(
        Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep",
        np.array(list_keep),
    )

    np.save(
        Path(__file__).parents[4] / "data/vattenfall/ams_pipe_diameters",
        np.array(pipe_dia_ams),
    )


def load_edge_svg():
    def find_css(string, keyword):
        start = string.find(keyword) + len(keyword) + 2
        end = string.find('"', start)
        return string[start:end]

    def convert_to_lines_svg(coo_string):
        coo_string = str(coo_string)
        start = 0
        end = len(coo_string) - 1
        coo_string = coo_string[start:end].split(" ")
        coo_string = [s.split(",") for s in coo_string if len(s) > 0]
        coo = np.array(coo_string, dtype=float)
        return coo

    def transform_svg_to_meter(svg_coos):
        svg_coos[:, 0] += 124000
        svg_coos[:, 1] += 480000
        return svg_coos

    path = Path(__file__).parents[4] / "data/vattenfall/edge.svg"

    edges_coordinates = []
    edges_dia = []
    with open(path, "r") as file:
        lines = file.readlines()
        classes = {}
        start_pharsing = False
        line_cach = ""
        for line in lines:
            if ".st" in line:
                key = line[line.find(".st") + 1 : line.find("{")]
                value_start = line.find("stroke-width:") + len("stroke-width:")
                if line.find("stroke-width:") == -1:
                    classes[key] = 1
                else:
                    value_end = line.find(";", value_start)
                    value = float(line[value_start:value_end])
                    classes[key] = value
            elif "</style>" in line:
                start_pharsing = True
            elif start_pharsing:
                if "</svg>" in line:
                    break
                if line[0] == "<":
                    line_cls = find_css(line, "class")
                    line_width = classes[line_cls] * 50

                if line.endswith("/>\n"):
                    line = line_cach + line
                    line_cach = ""
                    edges_dia.append(line_width)
                    if "<line" in line:
                        x1 = find_css(line, "x1")
                        y1 = find_css(line, "y1")
                        x2 = find_css(line, "x2")
                        y2 = find_css(line, "y2")
                        coos = np.array([[x1, y1], [x2, y2]], dtype=float)
                        edges_coordinates.append(transform_svg_to_meter(coos))

                    elif "<polyline" in line:
                        coos = convert_to_lines_svg(find_css(line, "points"))
                        edges_coordinates.append(transform_svg_to_meter(coos))
                else:
                    line_cach = line_cach + line[:-1]

    return edges_coordinates, edges_dia


def plot():

    list_keep = np.load(
        Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy"
    )
    rows = load_edge()
    coordinates = np.array([convert_to_lines(r[-4]) for r in rows])
    for edge in coordinates[list_keep]:
        edge = np.array(edge)
        plt.plot(edge[:, 1], edge[:, 0], c="b", linewidth=0.5)
    plt.show()


def edge_location_to_connection():
    # rows = load_edge()
    # list_keep = np.load(Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy")
    # rows = np.array(rows)[list_keep]
    # coordinates = [convert_to_lines(r[-4]) for r in rows]

    coordinates, _ = load_edge_svg()

    thres = 1
    connection_list = []
    for i, edge1 in enumerate(coordinates):
        print(i)
        for ii, coo1 in enumerate(edge1):
            for j, edge2 in enumerate(coordinates[i + 1 :]):
                for jj, coo2 in enumerate(edge2):
                    if np.array_equal(coo1, coo2):
                        assert np.array_equal(
                            coordinates[i][ii], coordinates[j + i + 1][jj]
                        )
                        connection_list.append([[i, ii], [j + i + 1, jj]])

    np.save(
        Path(__file__).parents[4] / "data/vattenfall/edge_connections_ams",
        np.array(connection_list),
    )


def check_connectivity():
    rows = load_edge()
    list_keep = np.load(
        Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy"
    )
    rows = np.array(rows)[list_keep]
    coordinates = [convert_to_lines(r[-4]) for r in rows]
    connection_list = np.load(
        Path(__file__).parents[4] / "data/vattenfall/edge_connections_ams.npy"
    )
    bool_visited = np.zeros(len(rows), dtype=bool)
    iter = 0

    while np.sum(bool_visited) < len(rows):
        start_idx = np.argmin(bool_visited)
        bool_visited[start_idx] = True
        explored_list = []
        exploring_list = [start_idx]

        while len(exploring_list) > 0:
            exploring_idx = exploring_list.pop()
            explored_list.append(exploring_idx)
            row_indices, col_indices = np.where(
                connection_list[:, :, 0] == exploring_idx
            )
            for r_idx, c_idx in zip(row_indices, col_indices):
                expend_idx = connection_list[r_idx, int(1 - c_idx), 0]
                if (expend_idx not in explored_list) & (
                    expend_idx not in exploring_list
                ):
                    bool_visited[expend_idx] = True
                    exploring_list.append(expend_idx)

        print(iter, np.sum(bool_visited), len(rows))
        iter += 1


def connect_edge_to_node():
    # rows = load_edge()
    # list_keep = np.load(Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy")
    # rows = np.array(rows)[list_keep]
    # edges_list = rows

    # edges_coordinates = [convert_to_lines(r[-4]) for r in rows]
    # geo_trans = Geo_trans()
    # edges_coordinates = [[geo_trans.to_meter(c[0], c[1]) for c in coos] for coos in edges_coordinates]
    edges_coordinates, _ = load_edge_svg()
    connection_list = np.load(
        Path(__file__).parents[4] / "data/vattenfall/edge_connections_ams.npy"
    )

    nodes_list = load_node()
    nodes_coo = [convert_to_point(n[-1]) for n in nodes_list]
    # for node connection, slot is on which coo of edge the node is connected to
    nodes_connections = []
    for i, n_coo in enumerate(nodes_coo[0:]):
        d_min = np.inf
        for j, edge_coordinates in enumerate(edges_coordinates):
            e_coo1 = edge_coordinates[0]
            e_coo2 = edge_coordinates[-1]
            d1 = np.sqrt(np.sum(np.power(n_coo - e_coo1, 2)))
            d2 = np.sqrt(np.sum(np.power(n_coo - e_coo2, 2)))
            # if min(d1,d2) < 10:
            #     print(min(d1,d2))
            if (d1 < d_min) or (d2 < d_min):
                # assert not ((min(d1, d2) < 10) & (d_min_sup < 10))
                d_min = min(d1, d2)
                if d1 < d2:
                    node_conenctions = [j, 0]
                else:
                    node_conenctions = [j, len(edge_coordinates) - 1]

        nodes_connections.append(node_conenctions)

    node_remove_indices = np.zeros(len(nodes_list), dtype=bool)
    edge_connection_list = connection_list.reshape(-1, 2)

    # remove nodes that are not connected to a end of a pipe
    # also remove certain node we don't want
    # for storage node, add extra connection to new dict
    extra_node_connections = {}
    for i, node in enumerate(nodes_list):
        if (node[4] == "WI0159") or (node[4] == "IW4301B"):
            node_remove_indices[i] = True
        else:
            node_connections = nodes_connections[i]
            if (
                np.sum(
                    (edge_connection_list[:, 0] == node_connections[0])
                    & (edge_connection_list[:, 1] == node_connections[1])
                )
                != 0
            ):
                # print(node[4])
                # if node[4] == "RK2300":
                #     x = np.where(
                #         (edge_connection_list[:,0] == node_connections[0])
                #         &(edge_connection_list[:,1] == node_connections[1])
                #     )[0]
                #     idx1, idx2 = connection_list[x[0]//2]
                #     print(x[0]//2)
                #     print(idx1, idx2)
                #     print(node[4],nodes_coo[i],
                #         edges_coordinates[node_connections[0]], node_connections[1])
                #     print(edges_coordinates[idx1[0]])
                #     print(edges_coordinates[idx2[0]])
                #     exit()
                if node[4] != "WI4301":
                    node_remove_indices[i] = True
                else:
                    x = np.where(
                        (edge_connection_list[:, 0] == node_connections[0])
                        & (edge_connection_list[:, 1] == node_connections[1])
                    )[0]
                    idx1, idx2 = connection_list[x[0] // 2]
                    extra_node_connections[i] = [list(idx1), list(idx2)]

    nodes_coo = np.array(nodes_coo)[np.invert(node_remove_indices)]
    nodes_connections = np.array(nodes_connections)[np.invert(node_remove_indices)]
    nodes_list = np.array(nodes_list)[np.invert(node_remove_indices)]

    # remove multiple nodes that are connected to a single pipe
    unique_connections, unique_connection_indices = np.unique(
        nodes_connections, axis=0, return_index=True
    )
    node_remove_indices2 = np.ones(len(nodes_list), dtype=bool)
    node_remove_indices2[unique_connection_indices] = False

    nodes_coo = np.array(nodes_coo)[np.invert(node_remove_indices2)]
    nodes_connections = np.array(nodes_connections)[np.invert(node_remove_indices2)]
    nodes_list = np.array(nodes_list)[np.invert(node_remove_indices2)]

    # after removing nodes, correct the extra node connections as well
    extra_node_connections_new = {}
    for key, value in extra_node_connections.items():
        assert node_remove_indices[key] == 0
        key = key - np.sum(node_remove_indices[:key])
        assert node_remove_indices2[key] == 0
        key = key - np.sum(node_remove_indices2[:key])
        extra_node_connections_new[key] = value

    return nodes_list, nodes_coo, nodes_connections, extra_node_connections_new


def find_splits():
    # rows = load_edge()
    # list_keep = np.load(Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy")
    # rows = np.array(rows)[list_keep]
    # coordinates = np.array([convert_to_lines(r[-4]) for r in rows], dtype=object)
    coordinates, _ = load_edge_svg()
    end_points = np.array([[c[0], c[-1]] for c in coordinates]).reshape((-1, 2))

    unique_points, indices_inverse, unique_counts = np.unique(
        end_points, return_inverse=True, return_counts=True, axis=0
    )

    split_coo = unique_points[unique_counts > 2]

    split_edge_connections = np.array(
        [np.where(indices_inverse == i)[0] for i in np.where(unique_counts > 2)[0]],
        dtype=object,
    )
    assert np.array_equal(
        np.array([len(x) for x in split_edge_connections]),
        unique_counts[unique_counts > 2],
    )
    # split_edge_connections = np.array([split_edge_connections//2, split_edge_connections%2])
    # split_edge_connections = np.transpose(split_edge_connections,(1,2,0))

    split_edge_connections2 = []
    for split_edge_connection in split_edge_connections:
        l = []
        for s in split_edge_connection:
            l.append([s // 2, -s % 2])
        split_edge_connections2.append(l)

    return split_coo, split_edge_connections2


def find_direct_edge_connections():
    coordinates, _ = load_edge_svg()
    end_points = np.array([[c[0], c[-1]] for c in coordinates]).reshape((-1, 2))

    unique_points, indices_inverse, unique_counts = np.unique(
        end_points, return_inverse=True, return_counts=True, axis=0
    )

    node_coo = unique_points[unique_counts == 2]
    node_edge_connections = np.array(
        [np.where(indices_inverse == i)[0] for i in np.where(unique_counts == 2)[0]],
        dtype=object,
    )
    assert np.array_equal(
        np.array([len(x) for x in node_edge_connections]),
        unique_counts[unique_counts == 2],
    )
    # split_edge_connections = np.array([split_edge_connections//2, split_edge_connections%2])
    # split_edge_connections = np.transpose(split_edge_connections,(1,2,0))

    node_edge_connections2 = []
    for node_edge_connection in node_edge_connections:
        l = []
        for s in node_edge_connection:
            l.append([s // 2, -s % 2])
        node_edge_connections2.append(l)

    return node_coo, node_edge_connections2


def edge_save_svg():
    rows = load_edge()
    list_keep = np.load(
        Path(__file__).parents[4] / "data/vattenfall/ams_indices_keep.npy"
    )
    edges_dia = np.load(
        Path(__file__).parents[4] / "data/vattenfall/ams_pipe_diameters.npy"
    )
    rows = np.array(rows)[list_keep]
    edges_list = rows
    edges_coordinates = [convert_to_lines(r[-4]) for r in rows]

    geo_trans = Geo_trans()

    def formatting(coos, width):
        head = '<polyline class="st1" points="'
        tail = '"/>\n'
        mid = ' " stroke-width="'
        coo_str = ""
        for coo in coos:
            coo = geo_trans.to_meter(coo[0], coo[1])
            coo = [round(coo[0] - 124000, 1), round(coo[1] - 480000, 1)]
            coo_str = coo_str + str(coo[0]) + "," + str(coo[1]) + " "

        dia_str = str(width / 50)

        return head + coo_str + mid + dia_str + tail

    saving_data = ""
    for edge_coordinates, edge_ida in zip(edges_coordinates, edges_dia):
        saving_str = formatting(edge_coordinates, edge_ida[0])
        saving_data += saving_str

    with open(Path(__file__).parents[4] / "data/vattenfall/edge_semi_svg", "w") as file:
        file.write(saving_data)


def plot_grid():
    edges_coordinates, edges_dia = load_edge_svg()
    connection_list = np.load(
        Path(__file__).parents[4] / "data/vattenfall/edge_connections_ams.npy"
    )

    splits_coo, split_edge_connections = find_splits()
    empty_node_coo, empty_node_edge_connections = find_direct_edge_connections()
    # splits_coo = np.array([geo_trans.to_meter(c[0], c[1]) for c in splits_coo])
    nodes_list, nodes_coo, node_edge_connection, _ = connect_edge_to_node()
    print(len(nodes_coo), len(splits_coo), len(empty_node_coo))
    import matplotlib.pyplot as plt

    plt.scatter(
        np.array(nodes_coo)[:, 0],
        np.array(nodes_coo)[:, 1],
        c="b",
        s=10,
        label="node",
        alpha=0.3,
    )
    plt.scatter(splits_coo[:, 0], splits_coo[:, 1], c="g", s=10, label="split")
    # plt.scatter([125514.7, 125524.2], [ 481560.5,  481704.], c='g', s=16, label = "split")
    plt.scatter(empty_node_coo[:, 0], empty_node_coo[:, 1], c="r", s=8, label="split")
    for edge_coordinates, edge_dia in zip(edges_coordinates, edges_dia):
        ec = np.array(edge_coordinates)
        # plt.plot(ec[:,0], ec[:,1], c='black', alpha=0.5, linewidth=edge_dia/200)
        plt.plot(ec[:, 0], ec[:, 1], c="black", alpha=0.5, linewidth=0.2)

    plt.legend()
    plt.show()
    # plt.axis('off')
    # # plt.gca().set_position([0, 0, 1, 1])
    # plt.savefig("test.svg")


def reverse_connection():
    edges_coordinates, edges_dia = load_edge_svg()
    connection_list = np.load(
        Path(__file__).parents[4] / "data/vattenfall/edge_connections_ams.npy"
    )

    splits_coo, split_edge_connections = find_splits()
    empty_node_coo, empty_node_edge_connections = find_direct_edge_connections()
    # splits_coo = np.array([geo_trans.to_meter(c[0], c[1]) for c in splits_coo])
    (
        nodes_list,
        nodes_coo,
        node_edge_connection,
        node_edge_connection_extra,
    ) = connect_edge_to_node()

    # edge_to_node_connection = -np.ones((len(edges_coordinates),2,2))
    edge_to_node_connection = [[] for _ in range(len(edges_coordinates))]
    producer_idx = np.where(nodes_list[:, 4] == "WI4300")[0][0]
    storage_idx = np.where(nodes_list[:, 4] == "WI4301")[0][0]

    for node_idx, (edge_idx, _) in enumerate(node_edge_connection):
        if node_idx != storage_idx:
            edge_to_node_connection[edge_idx].append([node_idx, -1])
        else:

            for edge_idx, _ in node_edge_connection_extra[node_idx]:
                edge_to_node_connection[edge_idx].append([node_idx, -1])

    for node_idx, empty_node_edge_connection in enumerate(empty_node_edge_connections):
        for edge_idx, _ in empty_node_edge_connection:
            edge_to_node_connection[edge_idx].append(
                [node_idx + len(nodes_list) + len(splits_coo), -1]
            )

    for node_idx, split_edge_connection in enumerate(split_edge_connections):
        for edge_idx, _ in split_edge_connection:
            edge_to_node_connection[edge_idx].append([node_idx + len(nodes_list), -1])

    # remove empty node that is at the same location as storage
    edge_indices_remove = []
    for i, e_n_c in enumerate(edge_to_node_connection):
        n_indices = [e_n[0] for e_n in e_n_c]
        if storage_idx in n_indices:
            assert len(e_n_c) == 3
            edge_indices_remove.append(i)

    assert len(edge_indices_remove) == 2
    n_indices = np.append(
        edge_to_node_connection[edge_indices_remove[0]],
        edge_to_node_connection[edge_indices_remove[1]],
    )
    uni_n_indices, counts = np.unique(n_indices, return_counts=True)
    remove_idx = uni_n_indices[(counts == 2) & (uni_n_indices != storage_idx)][0]
    edge_to_node_connection[edge_indices_remove[0]].remove([remove_idx, -1])
    edge_to_node_connection[edge_indices_remove[1]].remove([remove_idx, -1])
    empty_node_coo = np.delete(
        empty_node_coo, remove_idx - len(nodes_coo) - len(splits_coo), axis=0
    )
    empty_node_edge_connections = np.delete(
        empty_node_edge_connections,
        remove_idx - len(nodes_coo) - len(splits_coo),
        axis=0,
    )
    edge_to_node_connection = np.array(edge_to_node_connection)
    edge_to_node_connection[edge_to_node_connection[:, :, 0] > remove_idx] -= 1

    expending_node_edge_list = [[producer_idx, -1]]
    edges_connected_through_empty_nodes = {}
    # idx goes in this order: first producer&consumers& storage
    # then splits, finally empty nodes
    while len(expending_node_edge_list) > 0:
        node_idx, p_edge_idx = expending_node_edge_list.pop()
        # for all consumers
        if (
            (node_idx < len(nodes_list))
            & (node_idx != producer_idx)
            & (node_idx != storage_idx)
        ):
            continue
        # for producer
        elif node_idx == producer_idx:
            connected_edge_idx_and_slot = node_edge_connection[node_idx]
            edge_idx, _ = connected_edge_idx_and_slot
            upstream_idx = np.where(
                edge_to_node_connection[edge_idx, :, 0] == node_idx
            )[0][0]
            assert upstream_idx in [0, 1]
            edge_to_node_connection[edge_idx, upstream_idx, 1] = 0
            edge_to_node_connection[edge_idx, 1 - upstream_idx, 1] = 1
            downsteam_node_idx = edge_to_node_connection[edge_idx, 1 - upstream_idx, 0]
            expending_node_edge_list.append([downsteam_node_idx, edge_idx])
        # for all splits
        elif (node_idx < len(splits_coo) + len(nodes_list)) & (node_idx != storage_idx):
            for edge_idx, _ in split_edge_connections[node_idx - len(nodes_list)]:
                if edge_idx != p_edge_idx:
                    upstream_idx = np.where(
                        edge_to_node_connection[edge_idx, :, 0] == node_idx
                    )[0][0]
                    edge_to_node_connection[edge_idx, upstream_idx, 1] = 0
                    edge_to_node_connection[edge_idx, 1 - upstream_idx, 1] = 1
                    downsteam_node_idx = edge_to_node_connection[
                        edge_idx, 1 - upstream_idx, 0
                    ]
                    expending_node_edge_list.append([downsteam_node_idx, edge_idx])

        # for storage and empty nodes
        else:
            if node_idx == storage_idx:
                connected_edges_idx_and_slot = node_edge_connection_extra[node_idx]
            else:
                connected_edges_idx_and_slot = empty_node_edge_connections[
                    node_idx - len(nodes_coo) - len(splits_coo)
                ]
            [edge1_idx, _], [edge2_idx, _] = connected_edges_idx_and_slot
            if edge1_idx == p_edge_idx:
                n_edge_idx = edge2_idx
            else:
                assert edge2_idx == p_edge_idx
                n_edge_idx = edge1_idx

            if node_idx != storage_idx:
                edges_connected_through_empty_nodes[p_edge_idx] = n_edge_idx

            upstream_idx = np.where(
                edge_to_node_connection[n_edge_idx, :, 0] == node_idx
            )[0][0]

            edge_to_node_connection[n_edge_idx, upstream_idx, 1] = 0
            edge_to_node_connection[n_edge_idx, 1 - upstream_idx, 1] = 1
            downsteam_node_idx = edge_to_node_connection[
                n_edge_idx, 1 - upstream_idx, 0
            ]
            expending_node_edge_list.append([downsteam_node_idx, n_edge_idx])

    assert np.sum(edge_to_node_connection[:, :, 1] < 0) == 0
    assert np.all(np.sum(edge_to_node_connection[:, :, 1], axis=1) == 1)
    unique_nodes_indices, counts = np.unique(
        edge_to_node_connection[:, :, 0].flatten(), return_counts=True
    )
    assert np.array_equal(
        np.sort(np.append(unique_nodes_indices[counts == 1], storage_idx)),
        np.arange(len(nodes_coo)),
    )
    assert np.array_equal(
        np.sort(unique_nodes_indices[counts > 2]),
        np.arange(len(nodes_coo), len(nodes_coo) + len(splits_coo)),
    )
    assert np.array_equal(
        np.sort(unique_nodes_indices[counts == 2]),
        np.append(
            storage_idx,
            np.arange(
                len(nodes_coo) + len(splits_coo),
                len(nodes_coo) + len(splits_coo) + len(empty_node_coo),
            ),
        ),
    )

    edge_to_node_connection_simp = []
    for e_n_c in edge_to_node_connection:
        edge_to_node_connection_simp.append(e_n_c[np.argsort(e_n_c[:, 1]), 0])
    edge_to_node_connection_simp = np.array(edge_to_node_connection_simp)

    return (
        [edges_coordinates, edges_dia, connection_list],
        [splits_coo, split_edge_connections],
        [empty_node_coo, empty_node_edge_connections],
        [nodes_list, nodes_coo, node_edge_connection, node_edge_connection_extra],
        [[producer_idx], [storage_idx]],
        edge_to_node_connection_simp,
        edges_connected_through_empty_nodes,
    )


def get_edge_params(edges_coordinates, edges_dia):
    lengths = []
    Ks = []
    Vs_max = []
    pipe_params_loader = Pipe_params_loader()
    for edge_coordinates, dia in zip(edges_coordinates, edges_dia):
        Ks.append(pipe_params_loader.get_k(dia / 1000))
        Vs_max.append(pipe_params_loader.get_vmax(dia / 1000))
        length = 0
        for i in range(len(edge_coordinates) - 1):
            x1, y1 = edge_coordinates[i]
            x2, y2 = edge_coordinates[i + 1]
            length += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        lengths.append(length)

    return np.array(lengths), np.array(Ks), np.array(Vs_max)


class DataProcessor(object):
    """
    all coordinates are in Amersfoort coordinates (EPSG:7415), the distance can directly be calculated in meter
    edges_coordinates: coordinates of all pin points of edges
    edges_dia: diameter of the edge
    splits_coo: splits are what we call branch/junction in the simulator
    split_edge_connections: indexed the same as 'splits_coo'.
    for each split, it contains an array of [connected edge index, slot of the split that the edge is connected to]
    similarly goes for empty nodes and node
    empty node: a node that connects one edge to another without doing anything.
    While building the grid in the simulator it is recommended to avoid using empty nodes,
    instead, convert them to direct edge connection to avoid very short pipes.
    node: consumer, producer, storage
    node_edge_connection_extra: except storage, all node type is connected to only one edge,
    the storage is connected to two, thus it has an extra connection
    edge_to_node_connection: each edge is connected to 2 nodes. The shape of this is (num of edges, 2)
    The first value on the second axis is the index of the upstream node referring to.
    The indices are sorted starting from nodes to splits to empty nodes,
    for example, there are 100 nodes, 98 splits and 40 empty nodes,
    idx=2 refer to the 3rd node while idx=120 refer to the 20th split.
    The second value on the second axis is the downstream node.
    edges_connected_through_empty_nodes: dictionary of what's the downstream of an edge
    that is connected through an empty node. Notice this dict is directional
    """

    (
        [edges_coordinates, edges_dia, connection_list],
        [splits_coo, split_edge_connections],
        [empty_node_coo, empty_node_edge_connections],
        [nodes_list, nodes_coo, node_edge_connection, node_edge_connection_extra],
        [producers_indices, storage_indices],
        edge_to_node_connection,
        edges_connected_through_empty_nodes,
    ) = reverse_connection()

    """
    get parameters for pipes
    """
    edges_len, edges_K, edges_Vmax = get_edge_params(edges_coordinates, edges_dia)


# edge_location_to_connection()
# plot_grid()
# reverse_connection()
# DataProcessor()

def calculate_avg_distance_to_storage():
    (
        [edges_coordinates, edges_dia, connection_list],
        [splits_coo, split_edge_connections],
        [empty_node_coo, empty_node_edge_connections],
        [nodes_list, nodes_coo, node_edge_connection, node_edge_connection_extra],
        [producers_indices, storage_indices],
        edge_to_node_connection,
        edges_connected_through_empty_nodes,
    ) = reverse_connection()

    edges_len, edges_K, edges_Vmax = get_edge_params(edges_coordinates, edges_dia)

    producer_idx = np.where(nodes_list[:, 4] == "WI4300")[0][0]
    storage_idx = np.where(nodes_list[:, 4] == "WI4301")[0][0]

    distances = []
    for count, node_idx in enumerate(range(len(nodes_coo))):
        if (node_idx != producer_idx) & (node_idx != storage_idx):
            edge_idx, _ = node_edge_connection[node_idx]
            upstream_node_idx = edge_to_node_connection[edge_idx, 0]
            d = edges_len[edge_idx]
            while upstream_node_idx >= len(nodes_coo):
                node_idx = upstream_node_idx
                if node_idx < len(nodes_coo):
                    edge_idx, _ = node_edge_connection[node_idx]
                elif node_idx < (len(nodes_coo) + len(splits_coo)):
                    e_c = split_edge_connections[node_idx-len(nodes_coo)]
                    edge_idx = None
                    for e_idx, _ in e_c:
                        if edge_to_node_connection[e_idx, 1] == node_idx:
                            edge_idx = e_idx
                    assert edge_idx is not None
                else:
                    e_c = empty_node_edge_connections[node_idx-len(nodes_coo)-len(splits_coo)]
                    edge_idx = None
                    for e_idx, _ in e_c:
                        if edge_to_node_connection[e_idx, 1] == node_idx:
                            edge_idx = e_idx
                    assert edge_idx is not None
                upstream_node_idx = edge_to_node_connection[edge_idx, 0]

                d += edges_len[edge_idx]

            print(count, d)
            distances.append(d)

    print(np.mean(distances))

if __name__ == '__main__':
    plot_grid()
    # calculate_avg_distance_to_storage()

