import pandas as pd
import igraph as ig

class ex2:
    def __init__(self, data):
        # Initialize with the provided data
        self.data = data.melt(id_vars=['sqid', 'aid', 'sschlcde', 's1', 's2', 's12', 's18', 's44a1', 's44a2',
                                       's44a3', 's44a4', 's44a5', 's44a6', 's44a7', 's44a8', 's44a9', 's44a10',
                                       's44a11', 's44a12', 's44a13', 's44a14', 's44a15', 's44a16', 's44a17',
                                       's44a18', 's44a19', 's44a20', 's44a21', 's44a22', 's44a23', 's44a24',
                                       's44a25', 's44a26', 's44a27', 's44a28', 's44a29', 's44a30', 's44a31',
                                       's44a32', 's44a33', 's6a', 's6b', 's6c', 's6d', 's6e'], value_name='to_node_id')
        
        # Remove rows whose to_node_id is not in aid column, as far as these are friends from outside school and we just don't know their 'features'
        self.data = self.data[self.data['to_node_id'].isin(self.data['aid'].tolist())]
        
        # Create activity variable, summing all activities
        self.data['activity'] = self.data[['s44a1', 's44a2', 's44a3', 's44a4', 's44a5', 's44a6', 's44a7', 's44a8',
                                           's44a9', 's44a10', 's44a11', 's44a12', 's44a13', 's44a14', 's44a15',
                                           's44a16', 's44a17', 's44a18', 's44a19', 's44a20', 's44a21', 's44a22',
                                           's44a23', 's44a24', 's44a25', 's44a26', 's44a27', 's44a28', 's44a29',
                                           's44a30', 's44a31', 's44a32', 's44a33']].sum(axis=1)
        
        # Remove duplicate rows based on 'aid' and 'to_node_id'
        self.data.drop_duplicates(subset=['aid','to_node_id'], inplace=True)

        # Initialize an empty graph
        self.graph = ig.Graph()

        # Initialize graphs to None
        self.graphs = None

        # Initialize an empty DataFrame for one school's data
        self.data_one_school = pd.DataFrame()

    def generate_one_graph(self, school_code):
        # Filter data for the given school code
        data_one_school = self.data[self.data['sschlcde'] == school_code]
        
        # Create a graph from the filtered data
        self.graph = ig.Graph.TupleList(data_one_school[["aid", "to_node_id"]].itertuples(index=False), directed=False)
        
        # Assign activity values to nodes in the graph
        self.graph.vs['activity'] = data_one_school.merge(pd.Series(self.graph.vs()['name']).to_frame(), 
                                                          how='right', left_on='aid', right_on=0)['activity']
        
        # Extract the giant connected component of the graph
        self.graph = self.graph.connected_components(mode='weak').giant()
        
        return self.graph
        
    def generate_all_graphs(self):
        # Group data by school code and apply the _create_graph function to each group
        grouped_df = self.data.groupby('sschlcde', as_index=False)  # as_index=False to exclude the grouping columns
        self.graphs = grouped_df.apply(self._create_graph).reset_index(drop=True)
        return self.graphs

    @staticmethod # could be called without     
    def _create_graph(group):
        # Create a graph from the group data
        g = ig.Graph.TupleList(group[["aid", "to_node_id"]].itertuples(index=False), directed=False)
        
        # Assign activity values to nodes in the graph
        g.vs['activity'] = group.merge(pd.Series(g.vs()['name']).to_frame(), how='right', left_on='aid', 
                                       right_on=0)['activity']
        
        # Extract the giant connected component of the graph
        g = g.connected_components(mode='weak').giant()
        
        return g

    def correct_activity(self):
        # Correct activity values for nodes in the graph
        for node in self.graph.vs:
            if node['activity'] <= 3:
                node['activity'] = node['activity']
            else:
                node['activity'] = 4.0
        return self.graph