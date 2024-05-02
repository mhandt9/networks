from scipy.sparse.linalg import eigs
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp

class ex1:
    def __init__(self, friends_raw, schools_raw):

        self.data_raw = pd.merge(schools_raw, friends_raw, on='sqid')  # merge raw data

        # placehodlers
        self.data = pd.DataFrame()  # cleaned data
        self.school_graphs = {}  # school graphs
        self.eigenvalues = {}  # largest eigenvalues
        self.filtered_schools = {}  # filtered schools
        self.bonacich = {}  # bonacich centrality
        self.aggregate_activities = {}

    def clean(self):
        
        self.data = self.data_raw.replace([77777777, 88888888, 99999999], np.nan) #replace with missing values
        self.data = self.data.dropna(subset=['aid']) # remove missing values in aid column
        return self.data

    def generate_graphs(self):

        for school_code, school_df in self.data.groupby('sschlcde'):
            
            # empty graph at first
            G_school = nx.Graph()
            for _, row in school_df.iterrows():
                
                G_school.add_node(row['aid'])
                
                edges = [(row['aid'], friend_id) for friend_id in row[['mf1aid', 'mf2aid', 'mf3aid', 'mf4aid', 'mf5aid', 
                                                                      'ff1aid', 'ff2aid', 'ff3aid', 'ff4aid', 'ff5aid']].dropna()]
                
                G_school.add_edges_from(edges)
            
            # find the largest connected component for each school
            largest_component = max(nx.connected_components(G_school), key=len)
            largest_component_graph = G_school.subgraph(largest_component)
            
            # add to dictionary
            self.school_graphs[school_code] = largest_component_graph

        return self.school_graphs

    def _compute_largest_eigenvalue(self, graph):

        adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
        # Convert adjacency matrix to sparse matrix with data type 'float64'
        adjacency_matrix_sparse = sp.csr_matrix(adjacency_matrix, dtype=np.float64)
        # Compute eigenvalues
        eigenvalues, _ = eigs(adjacency_matrix_sparse, k=1, which='LR')  # k=1 for largest eigenvalue
        largest_eigenvalue = max(eigenvalues)  # Extract the largest eigenvalue

        return largest_eigenvalue

    def compute_largest_eigenvalues_and_filter(self, beta):
        
        for school_code, graph in self.school_graphs.items():

            largest_eigenvalue = self._compute_largest_eigenvalue(graph) #call hidden method for each graph
            #print(largest_eigenvalue)
            # Filter out
            if beta < (1 / largest_eigenvalue): 
                self.filtered_schools[school_code] = {
                    'graph': graph,
                    'largest_eigenvalue': largest_eigenvalue
                }
                self.eigenvalues[school_code] = largest_eigenvalue
        
        return self.filtered_schools
    
    def compute_bonacich_centrality(self, beta):
        # Compute Bonacich centrality for filtered schools
        for school_code, school_data in self.filtered_schools.items():
            graph = school_data['graph']
            adjacency_matrix = nx.adjacency_matrix(graph)
            
            n = adjacency_matrix.shape[0]
            I = np.eye(n)  # Identity matrix
            C = np.linalg.inv(I - beta * adjacency_matrix) @ (beta * adjacency_matrix @ np.ones((adjacency_matrix.shape[0], 1)))
            bonacich_centrality = C.flatten()
            self.bonacich[school_code] = bonacich_centrality
        return self.bonacich
    
    def compute_activity(self, alpha=0.01, beta=0.1):
        # Compute the interior solution for filtered schools
        for school_code, school_data in self.filtered_schools.items():
            graph = school_data['graph']
            adjacency_matrix = nx.adjacency_matrix(graph)
            
            n = adjacency_matrix.shape[0]
            I = np.eye(n)  # Identity matrix
            
            # Compute (I - beta * G)^-1
            inverse_matrix = np.linalg.inv(I - beta * adjacency_matrix)

            # Compute x* = alpha * (I - beta * G)^-1 * 1
            ones_vector = np.ones((n, 1))
            activity = alpha * np.dot(inverse_matrix, ones_vector).flatten()

            # Store the interior solution
            school_data['activity'] = activity

        return self.filtered_schools
    
    def aggregate_activity(self, plot):
        # Calculate aggregate activity per school normalized by school size
        aggregate_activities = []  # Store aggregate activities for all schools

        for school_code, school_data in self.filtered_schools.items():
            activity = school_data.get('activity', None)
            if activity is not None:
                aggregate_activity = np.sum(activity) / len(activity)  # Normalize by school size
                school_data['aggregate_activity'] = aggregate_activity
                aggregate_activities.append(aggregate_activity)  # Append to the list

        if plot:
            # Plot the distribution of aggregate activity
            plt.figure(figsize=(8, 6))
            plt.hist(aggregate_activities, bins=50, color='orange', edgecolor='black')
            plt.title('Distribution of Aggregate Activity per School (Normalized by School Size)')
            plt.xlabel('Aggregate Activity per School (Normalized)')
            plt.ylabel('Frequency')
            plt.show()

        
    def find_median_school(self):
        #Unfortunately need to calculate agg activity again
        # Calculate aggregate activity per school normalized by school size
        aggregate_activities = []

        for school_code, school_data in self.filtered_schools.items():
            activity = school_data.get('activity', None)
            if activity is not None:
                aggregate_activity = np.sum(activity) / len(activity)  # Normalize by school size
                school_data['aggregate_activity'] = aggregate_activity
                aggregate_activities.append((aggregate_activity, school_code))  # Append to the list

        # Sort the aggregate activities
        sorted_aggregate_activities = sorted(aggregate_activities)

        # Find the median
        median_index = (len(sorted_aggregate_activities) // 2) - 1 # minus one because of the python indexing
        median_activity = sorted_aggregate_activities[median_index][0]

        # Find the school with the median aggregate activity
        median_school = None
        for activity, school_code in sorted_aggregate_activities:
            if activity == median_activity:
                median_school = school_code
                break

        return median_school
    
    def key_player(self, graph):
        node_to_loops = {}
        for source_node in graph.nodes():
            paths = []
            for target in graph.neighbors(source_node):
                # all paths to neighbors that do not go through the same node twice
                paths_to_neighbors = list(nx.all_simple_paths(graph, source=source_node, target=target))
                # only those that are longer than 2 (the starting node is included so when len(l)=2 it is not a loop just an edge to a neighbor)
                paths += [l + [source_node] for l in paths_to_neighbors if len(l) > 2]
            node_to_loops[source_node] = paths

        key_players = []
        key_player_c = 0

        for i, node in enumerate(graph.nodes()):
            if len(node_to_loops[node]) == 0:
                continue  # Skip nodes with no loops
            # compute according to formula c_i = B_i^2 / m_ii
            intercentrality = round((self.bonacich[i])**2 / len(node_to_loops[node]), 4) # take length of the list as the amount of loops
            if intercentrality >= key_player_c:
                if intercentrality > key_player_c:
                    key_players = []  # Clear the list if a higher intercentrality is found
                key_players.append(node)
                key_player_c = intercentrality
            print(f'Node: {node} intercentrality: {intercentrality}')

        return key_players