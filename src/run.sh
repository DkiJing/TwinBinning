#!/bin/bash
echo "Generate graph..."
python graph_generate.py
echo "Clustering on grpah..."
python graph_cluster.py
