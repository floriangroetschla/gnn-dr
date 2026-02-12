"""
Graph visualization utilities for CoRe-GD.

Provides functions to visualize graph layouts computed by the model.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import numpy as np
from typing import Optional


def get_node_colors(graph, default_color='lightblue'):
    """
    Get node colors based on graph class labels if available.
    
    Args:
        graph: PyG Data object with optional 'y' attribute for class labels
        default_color: Color to use if no class labels available
        
    Returns:
        Either a color string (if no labels) or array of class indices for colormap
    """
    if hasattr(graph, 'y') and graph.y is not None:
        # Use class labels for coloring
        # Convert to numpy if needed
        if isinstance(graph.y, torch.Tensor):
            return graph.y.cpu().numpy()
        return graph.y
    return default_color


def visualize_graph_layout(graph, pos, title="Graph Layout", config=None):
    """
    Create matplotlib figure of graph with given 2D positions.
    
    Supports optional class-based node coloring if graph has 'y' attribute.
    
    Args:
        graph: PyG Data object with edge_index and optional 'y' for class labels
        pos: Node positions tensor (N x 2) from model output
        title: Plot title
        config: Optional ValidationConfig for styling
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Set default config values
    node_size = config.node_size if config else 50
    edge_width = config.edge_width if config else 0.5
    figure_size = config.figure_size if config else 10
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    
    # Convert to NetworkX graph
    edge_index = graph.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(edge_index.T)
    
    # Convert positions to dictionary
    pos_array = pos.detach().cpu().numpy()
    pos_dict = {i: pos_array[i] for i in range(len(pos_array))}
    
    # Determine node colors
    node_colors = get_node_colors(graph, default_color='lightblue')
    
    # Determine if we're using a colormap or single color
    if isinstance(node_colors, str):
        # Single color for all nodes
        cmap = None
        vmin = None
        vmax = None
    else:
        # Use colormap for class labels
        cmap = plt.cm.tab10  # 10 distinct colors for MNIST classes
        vmin = 0
        vmax = 9  # MNIST has 10 classes (0-9)
    
    # Draw graph
    nx.draw_networkx_edges(
        G, pos_dict, ax=ax,
        edge_color='gray',
        width=edge_width,
        alpha=0.8
    )
    
    nx.draw_networkx_nodes(
        G, pos_dict, ax=ax,
        node_size=node_size,
        node_color=node_colors if not isinstance(node_colors, str) else 'lightblue',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8
    )
    
    # Add legend if using class colors
    if not isinstance(node_colors, str):
        # Create legend with class colors
        legend_elements = [
            mpatches.Patch(facecolor=cmap(i/10.0), label=f'Class {i}')
            for i in range(10)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def create_comparison_figure(graph, pos_pred, pos_gt=None, title="Graph Layout Comparison", config=None):
    """
    Create side-by-side comparison of predicted and ground truth layouts.
    
    Supports optional class-based node coloring if graph has 'y' attribute.
    
    Args:
        graph: PyG Data object with edge_index and optional 'y' for class labels
        pos_pred: Predicted node positions (N x 2)
        pos_gt: Optional ground truth positions (N x 2)
        title: Plot title
        config: Optional ValidationConfig for styling
    
    Returns:
        matplotlib.figure.Figure: The created figure with subplots
    """
    # Set default config values
    node_size = config.node_size if config else 50
    edge_width = config.edge_width if config else 0.5
    figure_size = config.figure_size if config else 10
    
    # Determine number of subplots
    num_plots = 2 if pos_gt is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(figure_size * num_plots, figure_size))
    
    if num_plots == 1:
        axes = [axes]
    
    # Convert to NetworkX
    edge_index = graph.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(edge_index.T)
    
    # Determine node colors
    node_colors = get_node_colors(graph, default_color='lightblue')
    
    # Determine if we're using a colormap or single color
    if isinstance(node_colors, str):
        cmap = None
        vmin = None
        vmax = None
        pred_node_color = 'lightblue'
        gt_node_color = 'lightgreen'
    else:
        cmap = plt.cm.tab10
        vmin = 0
        vmax = 9
        pred_node_color = node_colors
        gt_node_color = node_colors
    
    # Plot predicted layout
    pos_pred_array = pos_pred.detach().cpu().numpy()
    pos_pred_dict = {i: pos_pred_array[i] for i in range(len(pos_pred_array))}
    
    nx.draw_networkx_edges(
        G, pos_pred_dict, ax=axes[0],
        edge_color='gray',
        width=edge_width,
        alpha=0.8
    )
    
    nx.draw_networkx_nodes(
        G, pos_pred_dict, ax=axes[0],
        node_size=node_size,
        node_color=pred_node_color,
        cmap=cmap if not isinstance(node_colors, str) else None,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8
    )
    
    axes[0].set_title("Predicted Layout", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # Plot ground truth if available
    if pos_gt is not None:
        pos_gt_array = pos_gt.detach().cpu().numpy()
        pos_gt_dict = {i: pos_gt_array[i] for i in range(len(pos_gt_array))}
        
        nx.draw_networkx_edges(
            G, pos_gt_dict, ax=axes[1],
            edge_color='gray',
            width=edge_width,
            alpha=0.8
        )
        
        nx.draw_networkx_nodes(
            G, pos_gt_dict, ax=axes[1],
            node_size=node_size,
            node_color=gt_node_color,
            cmap=cmap if not isinstance(node_colors, str) else None,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8
        )
        
        axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        axes[1].set_aspect('equal')
    
    # Add legend if using class colors
    if not isinstance(node_colors, str):
        legend_elements = [
            mpatches.Patch(facecolor=cmap(i/10.0), label=f'Class {i}')
            for i in range(10)
        ]
        fig.legend(handles=legend_elements, loc='upper center', fontsize=8, ncol=10, bbox_to_anchor=(0.5, 0.98))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
