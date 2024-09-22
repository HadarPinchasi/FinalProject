import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import networkx as nx
from IPython.display import display


def extract_gene_data(file_name: str):
    """
    Extracts gene symbol and log fold-change values from a CSV file.

    Parameters:
    - file_name (str): Path to the CSV file.

    Returns:
    - DataFrame: A DataFrame containing the 'gene_symbol' and 'logFc' columns.
    """
    data_frame = pd.read_csv(file_name)
    return data_frame[['gene_symbol', 'logFc']]


def retrieve_condition_label(csv_file):
    """
    Retrieves the condition label from the CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file.

    Returns:
    - str: The condition name found in the second column.
    """
    df = pd.read_csv(csv_file)
    condition_label = df.iloc[0, 2]
    return condition_label


# Initialize node names and collect unique gene symbols across files
nodes = ['Fibcd1', 'peripheral artery disease', 'aetotic stenosis', 'hypertrophic cardiomyopathy',
         'Tachycardia', 'calcific aortic valve disease', '1 hour exercise', 'long term',
         'cardiopulmonary exercise', 'Resistance Exercise']
gene_set = set()
csv_files = glob('*.csv')

# Update gene_set with unique gene symbols from each file
for file in csv_files:
    df = pd.read_csv(file)
    gene_set.update(df.iloc[:, 0])

# Initialize an empty matrix with nodes as rows and unique gene symbols as columns
gene_matrix = pd.DataFrame(0.0, index=nodes, columns=list(gene_set))

all_gene_data = []

# Populate data for each file
for file in csv_files:
    gene_df = extract_gene_data(file)
    node_name = retrieve_condition_label(file)
    gene_df['row_name'] = node_name
    all_gene_data.append(gene_df)

# Combine data from all files into one DataFrame
combined_gene_data = pd.concat(all_gene_data)

# Create a pivot table for gene symbols and conditions
pivot_gene_data = combined_gene_data.pivot_table(index='row_name', columns='gene_symbol', values='logFc', fill_value=0)

# Align the pivot table with the structure of the initialized matrix
pivot_gene_data = pivot_gene_data.reindex(index=gene_matrix.index, columns=gene_matrix.columns, fill_value=0)
gene_matrix.update(pivot_gene_data)

# Save the resulting matrix to a CSV file
gene_matrix.to_csv('output_gene_matrix.csv')

# Filter out rows where all values are zero
non_zero_rows = (gene_matrix != 0).any(axis=1)
gene_matrix = gene_matrix[non_zero_rows]

# Normalize the matrix (adding a small epsilon to avoid division by zero)
epsilon = 1e-10
gene_matrix += epsilon

row_norms = np.linalg.norm(gene_matrix, axis=1)
row_norms[row_norms == 0] = 1  # Prevent division by zero
normalized_matrix = gene_matrix.div(row_norms, axis=0)

# Compute cosine similarity for normalized matrix
cos_similarity_matrix = cosine_similarity(normalized_matrix)

# Convert to DataFrame for readability
cos_similarity_df = pd.DataFrame(cos_similarity_matrix, index=gene_matrix.index, columns=gene_matrix.index)

# Replace diagonal values with NaN for clearer heatmap visualization
np.fill_diagonal(cos_similarity_df.values, np.nan)

# Plot the heatmap of cosine similarity
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(cos_similarity_df, annot=True, fmt=".3f", cmap="BuGn", cbar=True, linewidths=.5)
heatmap.set_title('Cosine Similarity between Condition Nodes', pad=12)
plt.show()

# Remove columns with fewer than 2 non-zero values
column_mask = (gene_matrix != 0).sum(axis=0) >= 2
filtered_gene_matrix = gene_matrix.loc[:, column_mask]

# Remove values within ± 2 * std deviation
std_deviation = filtered_gene_matrix.stack().std()
filtered_gene_matrix[(filtered_gene_matrix < 0) & (filtered_gene_matrix > -2 * std_deviation)] = 0
filtered_gene_matrix[(filtered_gene_matrix > 0) & (filtered_gene_matrix < +2 * std_deviation)] = 0

# Remove columns with fewer than 2 non-zero values again after filtering
filtered_gene_matrix = filtered_gene_matrix.loc[:, (filtered_gene_matrix != 0).sum(axis=0) >= 2]

# Normalize filtered matrix column-wise
column_norms = np.linalg.norm(filtered_gene_matrix, axis=0)
normalized_filtered_matrix = filtered_gene_matrix.div(column_norms, axis='columns')

# Calculate cosine similarity between gene columns
gene_cosine_similarity_matrix = cosine_similarity(normalized_filtered_matrix.T)

# Convert to DataFrame for readability
gene_similarity_df = pd.DataFrame(gene_cosine_similarity_matrix, index=filtered_gene_matrix.columns,
                                  columns=filtered_gene_matrix.columns)

# Create a graph from the condition-gene matrix
condition_list = filtered_gene_matrix.index.tolist()
gene_list = filtered_gene_matrix.columns.tolist()

graph = nx.Graph()

# Add condition and gene nodes
for condition in condition_list:
    graph.add_node(condition, node_type='condition')

for gene in gene_list:
    graph.add_node(gene, node_type='gene')

# Add edges between conditions and genes based on matrix values
for condition in condition_list:
    for gene in gene_list:
        weight = filtered_gene_matrix.at[condition, gene]
        if weight != 0:
            graph.add_edge(condition, gene, weight=weight)

# Graph properties: number of nodes and edges
print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())


# Example: Calculate Gene-Condition Prioritization Score (GCPS) for a condition
def calculate_gcps(condition, hetero_graph):
    """
    Calculate the Prioritization Score (GCPS) for a specific condition and its related genes.

    Parameters:
    - condition (str): The condition node for which the score is calculated.
    - hetero_graph (Graph): A graph with condition and gene nodes.

    Returns:
    - Series: Mean cosine similarity scores for related genes.
    """
    related_genes = list(hetero_graph.neighbors(condition))
    gene_cosine_subset = gene_similarity_df.reindex(columns=related_genes, fill_value=0)
    return gene_cosine_subset.mean(axis=1)


# Example usage of GCPS
gcps_example_score = calculate_gcps("1 hour exercise", graph).get("MGAM", 0)
print("GCPS score:", gcps_example_score)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import networkx as nx

def predict_new_edges(G, mlp_model, model, gene_gene_cos_sim_matrix):
    """
    Predicts new potential edges between conditions and genes that are not currently connected
    in the given graph using the trained MLP model and the node2vec embeddings.

    Parameters:
    -----------
    G : networkx.Graph
        The heterogeneous graph containing conditions and genes.
    mlp_model : sklearn.neural_network.MLPClassifier
        The trained MLP model used to predict the likelihood of edges between conditions and genes.
    model : node2vec model
        Trained node2vec model containing the embeddings for nodes in the graph.
    gene_gene_cos_sim_matrix : pandas.DataFrame
        Cosine similarity matrix between genes, used to compute GCPS (Gene Condition Prioritization Score).

    Returns:
    --------
    new_edges : list of tuples
        A list of predicted new edges in the format (condition, gene, prediction_score),
        sorted by the prediction score in descending order.
    """
    new_edges = []
    conditions = [node for node, data in G.nodes(data=True) if data['type'] == 'condition']
    genes = [node for node, data in G.nodes(data=True) if data['type'] == 'gene']

    for condition in conditions:
        for gene in genes:
            if not G.has_edge(condition, gene):
                # Check if both condition and gene exist in the node2vec model
                if condition in model.wv and gene in model.wv:
                    vector_u_v = model.wv[condition] + model.wv[gene]
                    gCps_score = GCPS(condition, G).get(gene, np.nan)
                    if np.isnan(gCps_score):
                        gCps_score = 0
                    feature_vector = np.append(vector_u_v, gCps_score)
                    prediction = mlp_model.predict_proba([feature_vector])[0][1]
                    if prediction > 0.5:  # Threshold for edge prediction (can be adjusted)
                        new_edges.append((condition, gene, prediction))

    return sorted(new_edges, key=lambda x: x[2], reverse=True)


# Usage example of predicting new edges
predicted_edges = predict_new_edges(G, mlp_model, model, gene_gene_cos_sim_matrix)
print(len(predicted_edges))

# Display the top 20 predicted edges
print("Top 20 predicted new edges:")
for condition, gene, probability in predicted_edges[:20]:
    print(f"{condition} - {gene}: {probability:.4f}")


def visualize_graph_with_new_edges(G, new_edges, top_n=20):
    """
    Visualizes the graph with the top predicted new edges.

    Parameters:
    -----------
    G : networkx.Graph
        The original heterogeneous graph with existing edges.
    new_edges : list of tuples
        A list of new edges predicted, with format (condition, gene, prediction_score).
    top_n : int
        The number of top predicted edges to visualize.

    Returns:
    --------
    None
    """
    G_new = G.copy()
    for condition, gene, _ in new_edges[:top_n]:
        G_new.add_edge(condition, gene, color='r', weight=2)

    pos = nx.spring_layout(G_new)
    plt.figure(figsize=(15, 10))

    # Draw original edges
    nx.draw_networkx_edges(G_new, pos, edge_color='gray', alpha=0.5)

    # Draw new predicted edges
    new_edges_for_draw = [(u, v) for u, v, _ in new_edges[:top_n]]
    nx.draw_networkx_edges(G_new, pos, edgelist=new_edges_for_draw, edge_color='r', width=2)

    # Draw nodes
    condition_nodes = [node for node, data in G_new.nodes(data=True) if data['type'] == 'condition']
    gene_nodes = [node for node, data in G_new.nodes(data=True) if data['type'] == 'gene']

    nx.draw_networkx_nodes(G_new, pos, nodelist=condition_nodes, node_color='skyblue', node_size=300, label='Conditions')
    nx.draw_networkx_nodes(G_new, pos, nodelist=gene_nodes, node_color='lightgreen', node_size=200, label='Genes')

    nx.draw_networkx_labels(G_new, pos, font_size=8)

    plt.title("Graph with Top 20 Predicted New Edges")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Visualize the top 20 predicted new edges
visualize_graph_with_new_edges(G, predicted_edges)


def plot_roc_curve(mlp_model, X_test, y_test):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for the trained MLP model on the test set.

    Parameters:
    -----------
    mlp_model : sklearn.neural_network.MLPClassifier
        The trained MLP model.
    X_test : numpy.ndarray
        Feature matrix of the test set.
    y_test : numpy.ndarray
        True labels of the test set.

    Returns:
    --------
    None
    """
    y_test_proba = mlp_model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and area under the curve (AUC)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Plot ROC curve for the MLP model on the test set
plot_roc_curve(mlp_model, X_test, y_test)
