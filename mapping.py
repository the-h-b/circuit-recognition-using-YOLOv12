from scipy.spatial import distance

def generate_netlist(detections, intersections, labels, component_labels, model):
    """Generates the LTSpice netlist."""
    netlist = []
    component_nodes = {}
    clusters = {}
    for i, (x, y) in enumerate(intersections):
        cluster_id = labels[len(detections) + i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append((x, y))

    for index, row in detections.iterrows():
        cluster_id = component_labels[index]
        if cluster_id in clusters:
            component_nodes[index] = clusters[cluster_id]

    node_mapping = {}
    node_counter = 1
    for cluster_id, nodes in clusters.items():
        node_mapping[cluster_id] = f"N{node_counter}"
        node_counter += 1

    for index, row in detections.iterrows():
        component_name = row['name']
        connected_nodes = component_nodes[index]

        if len(connected_nodes) >= 2:
            node1 = node_mapping[component_labels[index]]
            second_node_cluster = None
            for other_cluster in clusters:
                if other_cluster != component_labels[index]:
                    for node in clusters[other_cluster]:
                        for comp_node in connected_nodes:
                            if distance.euclidean(node, comp_node) < 5:
                                second_node_cluster = other_cluster
                                break
                    if second_node_cluster is not None:
                        break
            if second_node_cluster is None:
                continue

            node2 = node_mapping[second_node_cluster]

            # Basic mapping. Add more complex mappings for different components.

            if component_name == 'Resistor':

                netlist.append(f"R{index} {node1} {node2} 1k") # Default 1k resistor.

            elif component_name == 'Capacitor':

                netlist.append(f"C{index} {node1} {node2} 1u") # Default 1u capacitor.

            elif component_name == 'Inductor':

                netlist.append(f"L{index} {node1} {node2} 1m") # Default 1m inductor.

            elif component_name == 'DCSource':

                netlist.append(f"V{index} {node1} {node2} 5") # Default 5V source.

            elif component_name == 'Gnd':

                netlist.append(f"GND{index} {node1} 0") # Ground

            elif component_name == 'Diode':

                netlist.append(f"D{index} {node1} {node2} D1N4148")

            elif component_name == 'Ammeter':

                netlist.append(f"A{index} {node1} {node2} 0")

            elif component_name == 'Voltmeter':

                netlist.append(f"Vmeter{index} {node1} {node2} 0")

            elif component_name == 'ACSource':

              netlist.append(f"VAC{index} {node1} {node2} SIN(0 1 1k)")

            elif component_name == 'Cell':

              netlist.append(f"Cell{index} {node1} {node2} 1.5")

            elif component_name == 'DCcurrentsrc':

              netlist.append(f"I{index} {node1} {node2} 1m")

            elif component_name == 'DepSource':

              netlist.append(f"E{index} {node1} {node2} VALUE={{V({node1})-V({node2})}}") #Voltage dependent voltage source.

            elif component_name == 'DepcurrentSrc':

              netlist.append(f"G{index} {node1} {node2} VALUE={{I(V{index})}}") #Current dependent current source.

            elif component_name == 'NMOS':

              netlist.append(f"M{index} {node1} {node2} 0 NMOS L=1u W=10u") #Drain, gate, source

            elif component_name == 'PMOS':

              netlist.append(f"M{index} {node1} {node2} 0 PMOS L=1u W=10u") #Drain, gate, source

            elif component_name == 'NPN':

              netlist.append(f"Q{index} {node1} {node2} 0 NPN") #Collector, base, emitter

            elif component_name == 'PNP':

              netlist.append(f"Q{index} {node1} {node2} 0 PNP") #Collector, base, emitter

            elif component_name == 'AND':

              netlist.append(f"X{index} {node1} {node2} ANDGATE") #Logic gates need subcircuits.

            elif component_name == 'NAND':

              netlist.append(f"X{index} {node1} {node2} NANDGATE")

            elif component_name == 'NOT':

              netlist.append(f"X{index} {node1} {node2} NOTGATE")

            elif component_name == 'XOR':

              netlist.append(f"X{index} {node1} {node2} XORGATE")

    return netlist