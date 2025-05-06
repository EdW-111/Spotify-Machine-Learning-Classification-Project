# Spotify-Machine-Learning-Classification-Project

This project explores the classification of music genres using Spotify audio features. Working with a dataset of 50,005 songs evenly split across 10 genres, I applied end-to-end machine learning workflows to predict a song’s genre based on features like acousticness, tempo, energy, and more.
Key Features
	•	Data Cleaning: Imputed missing values using genre-wise medians; handled non-numeric entries and duplicate records.
	•	Feature Engineering: Applied one-hot encoding to categorical variables and square-root transformation to fix right-skewed distributions (e.g., acousticness).
	•	Dimensionality Reduction: Performed PCA to retain 95% of the variance and reduce model complexity.
	•	Models Implemented:
	•	Logistic Regression
	•	Random Forest
	•	SVM (RBF Kernel)
	•	Feedforward Neural Network (PyTorch)
Highlights
	•	Demonstrates the use of classical ML and deep learning in a multi-class classification setting
	•	Shows effective preprocessing, normalization, and transformation strategies
	•	Emphasizes interpretability vs. performance trade-offs in model selection
