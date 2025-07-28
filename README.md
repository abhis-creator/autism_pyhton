# Autism Therapy Engagement Analysis

A machine learning framework for analyzing autism therapy engagement patterns using unsupervised learning techniques on CHILDES and NDAR datasets.

## Overview

This project applies clustering and anomaly detection algorithms to identify patterns in autism therapy sessions, combining data from:
- **CHILDES** (Child Language Data Exchange System) transcripts
- **NDAR** (National Database for Autism Research) screening data

## Features

- **Data Processing**: Extracts therapy-related features from CHILDES transcripts and NDAR data
- **Clustering Analysis**: K-means and DBSCAN clustering to identify engagement patterns
- **Anomaly Detection**: Autoencoder-based anomaly detection for unusual therapy sessions
- **Visualization**: PCA-based cluster visualization and distance distribution plots
- **Comprehensive Reporting**: Detailed summaries and silhouette score analysis

## Project Structure

```
autism_pyhton/
├── engagement Therapy.py          # Main analysis script
├── data/
│   ├── Autism_Screening_Data_Combined.csv
│   └── childes_SLI.csv
├── cluster_plot.png              # K-means cluster visualization
├── dbscan_cluster_plot.png       # DBSCAN cluster visualization
├── distance_distribution.png     # Distance distribution for DBSCAN tuning
├── pca_coordinates.csv           # PCA coordinates with cluster labels
├── kmeans_summary.csv            # K-means cluster statistics
├── dbscan_summary.csv            # DBSCAN cluster statistics
├── anomaly_summary.csv           # Anomaly detection results
└── silhouette_scores.csv         # Model evaluation metrics
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd autism_pyhton
```

2. Install required dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib nltk scipy
```

3. Download NLTK data (automatically handled by the script):
```python
import nltk
nltk.download('punkt')
```

## Usage

Run the main analysis script:
```bash
python "engagement Therapy.py"
```

The script will:
1. Load data from the `data/` directory (or generate synthetic data if files are missing)
2. Preprocess the data with imputation, outlier filtering, and PCA
3. Apply K-means and DBSCAN clustering
4. Perform anomaly detection using autoencoders
5. Generate visualizations and summary reports

## Key Functions

- `process_childes_transcript()`: Extracts features from CHILDES transcript files
- `preprocess_data()`: Handles missing values, outliers, and dimensionality reduction
- `apply_unsupervised_learning()`: Implements clustering and anomaly detection
- `visualize_clusters()`: Creates PCA-based cluster visualizations

## Output Files

- **Visualizations**: PNG files showing cluster distributions and distance analysis
- **Summaries**: CSV files with cluster statistics and anomaly patterns
- **Coordinates**: PCA coordinates for further analysis
- **Metrics**: Silhouette scores and optimal DBSCAN parameters

## Data Requirements

### CHILDES Format
Transcript files should contain:
- `*CHI:` lines for child utterances
- `*ADU:` lines for adult utterances

### NDAR Format
CSV files should include columns for:
- Interaction frequency
- Session duration
- Engagement scores
- Repetitive behavior measures

## Results Interpretation

- **K-means clusters**: Identify distinct therapy engagement profiles
- **DBSCAN clusters**: Discover density-based patterns and outliers
- **Anomaly scores**: Highlight unusual therapy sessions requiring attention
- **Silhouette scores**: Evaluate clustering quality (higher is better)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please contact the project maintainer.

## Acknowledgments

- CHILDES database for providing transcript data
- NDAR for autism screening datasets
- scikit-learn and TensorFlow communities for machine learning tools
