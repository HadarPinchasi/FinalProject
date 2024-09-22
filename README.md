This code is designed to analyze relationships between biological conditions (such as diseases or physical exercises) and genes using graph-based machine learning techniques. The main purpose of the project is to predict potential new associations between conditions and genes that are not currently known, based on existing data.

# Key Features:
## Heterogeneous Graph Analysis:
The code operates on a heterogeneous graph where nodes represent both conditions (e.g., diseases, activities) and genes, and edges represent known relationships between them.

## Node Embeddings:
It uses a node2vec algorithm to generate embeddings (vector representations) for each node in the graph. These embeddings capture patterns and relationships between conditions and genes.

## MLP Classifier for Prediction:
The project leverages a Multi-Layer Perceptron (MLP) classifier to predict potential new edges (connections) between conditions and genes by training on known relationships.

## Cosine Similarity & GCPS:
It uses the Gene Condition Prioritization Score (GCPS), which is based on cosine similarity between gene vectors, to enhance the prediction of new edges.

## Predict and Visualize New Relationships:
After training, the model can predict new condition-gene pairs that may have not been previously discovered, and these predictions can be visualized in a graph.

## Model Evaluation:
The code includes tools to evaluate the performance of the model through metrics like accuracy and ROC curves, helping assess the reliability of the predictions.

This project is useful for researchers and data scientists working in bioinformatics, helping to uncover potentially significant relationships between genes and various biological conditions.

# How to make it work?

## Requirements
Python 3.7+

Install the required Python packages using pip:

bash
pip install -r requirements.txt
### Required libraries:

numpy
scikit-learn
matplotlib
networkx
node2vec or similar library for embedding nodes
pandas

### Project Structure
- predict_new_edges function: Predicts new potential connections (edges) between conditions and genes in the graph.
- visualize_graph_with_new_edges function: Visualizes the graph and highlights the top predicted new edges.
- plot_roc_curve function: Plots the Receiver Operating Characteristic (ROC) curve to evaluate the performance of the MLP model.
- Data: You will need a heterogeneous graph, a trained node2vec model, and a gene-gene cosine similarity matrix to run the predictions.


