import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp

class ex1():
    def __init__(self, friends_raw, schools_raw):

        self.friends_raw = friends_raw
        self.schools_raw = schools_raw
        self.data_raw = pd.merge(schools_raw, friends_raw, on='sqid')  # merge raw data

        # placehodlers
        self.data = pd.DataFrame()  # cleaned data
        self.school_graphs = {}  # school graphs
        self.eigenvalues = {}  # largest eigenvalues
        self.filtered_schools = []  # filtered schools
        self.bonacich = {}  # bonacich centrality

    def clean(self, id_list):
        
        # removing weird IDs from the raw data
        for col in self.data_raw:
            self.data_raw[col] = [np.nan if x in id_list else x for x in self.data_raw[col]]
        
        self.data_raw = self.data_raw[self.data_raw['aid'] != '']  # remove rows with empty 'aid'
        
        self.data = self.data_raw.dropna(subset=['sqid'])  # drop rows with missing 'sqid'
        
        return self.data

    def generate_graphs(self):
        
        for school_code, school_df in self.data.groupby('sschlcde'):
            
            # empty graph at first
            G_school = nx.Graph()
            for _, row in school_df.iterrows():
                
                # add nodes and edges to the graph based on named friends
                node_attributes = row.drop(['aid', 'sqid'])
                
                G_school.add_node(row['sqid'], **node_attributes.to_dict())
                
                edges = [(row['aid'], friend_id) for friend_id in row[['mf1aid', 'mf2aid', 'mf3aid', 'mf4aid', 'mf5aid', 
                                                                      'ff1aid', 'ff2aid', 'ff3aid', 'ff4aid', 'ff5aid']].dropna()]
                
                G_school.add_edges_from(edges)
            
            # find the largest connected component for each school
            largest_component = max(nx.connected_components(G_school), key=len)
            largest_component_graph = G_school.subgraph(largest_component)
            
            # add to dictionary
            self.school_graphs[school_code] = largest_component_graph

    def _compute_largest_eigenvalue(self, graph):

        # compute the largest eigenvalue of a graph's adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(graph).astype(np.float64)
        adjacency_matrix_sparse = sp.csr_matrix(adjacency_matrix) # much faster using sparse matrix
        
        return max(sp.linalg.eigsh(adjacency_matrix_sparse, k=1, which='LA', return_eigenvectors=False))

    def compute_eigenvalues_and_filter_schools(self, beta):
        # Compute largest eigenvalues and filter schools based on the condition
        for school_code, graph in self.school_graphs.items():
            largest_eigenvalue = self._compute_largest_eigenvalue(graph)
            if beta < 1 / largest_eigenvalue:
                self.filtered_schools.append({
                    'school_code': school_code,
                    'graph': graph,
                    'largest_eigenvalue': largest_eigenvalue
                })
                self.eigenvalues[school_code] = largest_eigenvalue
        return self.filtered_schools

    def compute_bonacich_centrality(self, beta=0.01):
        # Compute Bonacich centrality for filtered schools
        for school_data in self.filtered_schools:
            school_code = school_data['school_code']
            graph = school_data['graph']
            adjacency_matrix = nx.adjacency_matrix(graph)
            n = adjacency_matrix.shape[0]
            I = np.eye(n)  # Identity matrix
            C = np.linalg.inv(I - beta * adjacency_matrix.T) @ np.ones((n, 1))
            bonacich_centrality = C.flatten()
            self.bonacich[school_code] = bonacich_centrality
        return self.bonacich
