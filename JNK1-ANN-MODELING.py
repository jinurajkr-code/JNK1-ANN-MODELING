"""
JNK1 Artificial Neural Network (ANN) Regression Modeling Script             
================================================================
Developed by: KR JINURAJ and V.N. BALAJI

This script performs ANN regression modeling with:
- 80:20 train-test split
- 10-fold cross-validation
- Two architectures: Single hidden layer and Double hidden layers
- Single layer: neurons = 2/3 of features
- Double layer: first = 2/3 of features, second = 1/3 of features
- Evaluation metrics: Correlation coefficient, R-squared, MAE
- Visualization: True vs Predicted pIC50 plots

Purpose:
--------
To predict JNK1 protein activity (pIC50 values) using molecular descriptors
through artificial neural network regression modeling.

Methodology:
------------
1. Data Loading: Import molecular descriptor dataset from CSV
2. Data Preprocessing: Handle missing values, convert to numeric, clean data
3. Train-Test Split: 80% training, 20% testing with random_state=42
4. Feature Scaling: StandardScaler for normalization (mean=0, std=1)
5. Model Training: Two ANN architectures with MLPRegressor
6. Cross-Validation: 10-fold CV for robust performance estimation
7. Evaluation: Multiple metrics (R², Correlation, MAE) on train/CV/test sets
8. Visualization: Scatter plots with trend lines and perfect prediction reference
9. Model Persistence: Save best model, scaler, and metadata for future use
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd              # For data manipulation and CSV handling
import numpy as np               # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating visualizations and plots
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
                                # train_test_split: Split data into train/test sets
                                # cross_val_predict: Generate cross-validation predictions
                                # KFold: K-fold cross-validation splitter
from sklearn.neural_network import MLPRegressor
                                # Multi-layer Perceptron regressor for ANN modeling
from sklearn.preprocessing import StandardScaler
                                # Feature scaling to standardize input features
from sklearn.metrics import mean_absolute_error, r2_score
                                # Evaluation metrics for regression models
from scipy.stats import pearsonr # For calculating Pearson correlation coefficient
import joblib                    # For saving/loading trained models
import warnings                  # For warning message control
import os                        # For file system operations
warnings.filterwarnings('ignore') # Suppress warning messages for cleaner output

# ============================================================================
# FILE PATH CONFIGURATION
# ============================================================================

# Input CSV file containing molecular descriptors and pIC50 values
# Expected columns: 'Molecule ChEMBL ID', 'Smiles', 'Standard Value', 'pIC50', and molecular descriptors
input_file = r'I:\JNK1-VERSION-7\JNK1-DATASET-V12.csv'

# ============================================================================
# LOAD AND INSPECT DATASET
# ============================================================================

print("="*70)
print("JNK1 Artificial Neural Network (ANN) Regression Modeling")
print("Developed by: KR JINURAJ and V.N. BALAJI")
print("="*70)
print(f"\nReading file: {input_file}")

# Read CSV file with multiple encoding attempts to handle different file formats
# This ensures compatibility across different systems and file encodings
try:
    # First attempt: UTF-8 encoding (most common)
    df = pd.read_csv(input_file, encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    try:
        # Second attempt: Latin-1 encoding (ISO-8859-1)
        df = pd.read_csv(input_file, encoding='latin-1', low_memory=False)
    except UnicodeDecodeError:
        # Third attempt: Windows CP1252 encoding (common in Windows systems)
        df = pd.read_csv(input_file, encoding='cp1252', low_memory=False)

# Display basic dataset information for verification
print(f"Dataset shape: {df.shape}")          # (rows, columns)
print(f"Number of rows: {len(df)}")          # Total number of molecules
print(f"Number of columns: {len(df.columns)}") # Total features + metadata

# ============================================================================
# IDENTIFY FEATURES AND TARGET VARIABLE
# ============================================================================

# Define columns that are NOT features (metadata and target variable)
# These columns will be excluded from the feature matrix
original_cols = ['Molecule ChEMBL ID',  # Unique identifier for molecules
                 'Smiles',               # SMILES notation of molecular structure
                 'Standard Value',       # Raw activity value (IC50 in nM)
                 'pIC50']               # Target variable (negative log of IC50)

# Extract feature columns (all columns except metadata and target)
# These are the molecular descriptors that will be used for prediction
feature_cols = [col for col in df.columns if col not in original_cols]

print(f"\nFeature columns: {len(feature_cols)}")  # Number of molecular descriptors
print(f"Target variable: pIC50")                  # What we're trying to predict

# ============================================================================
# DATA PREPARATION AND CLEANING
# ============================================================================

# Separate features (X) and target variable (y)
# .copy() prevents SettingWithCopyWarning and ensures independent dataframes
X = df[feature_cols].copy()  # Independent variables (molecular descriptors)
y = df['pIC50'].copy()       # Dependent variable (activity)

print("\nPreparing features...")

# Convert all feature columns to numeric format
# 'coerce' converts non-numeric values to NaN (Not a Number)
# This handles any text or invalid entries in the descriptor columns
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Replace NaN values with 0 (missing descriptor values)
# Assumption: Missing descriptors are treated as zero contribution
# Alternative approaches: mean imputation, median imputation, or row deletion
X = X.fillna(0)

# Remove rows where target variable (pIC50) is missing
# These samples cannot be used for training or testing
valid_mask = ~pd.isna(y)  # Create boolean mask for valid rows (True where not NaN)
X = X[valid_mask]         # Keep only valid feature rows
y = y[valid_mask]         # Keep only valid target rows

print(f"Valid samples after cleaning: {len(X)}")

# ============================================================================
# SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================================

print("\n" + "="*70)
print("Splitting data: 80% training, 20% test")
print("="*70)

# Split data: 80% for training (model learning), 20% for testing (final evaluation)
# random_state=42 ensures reproducibility (same split every time)
# This is important for comparing results across different runs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")  # Samples used for model training
print(f"Test set size: {len(X_test)}")       # Samples held out for final testing

# ============================================================================
# FEATURE SCALING (STANDARDIZATION)
# ============================================================================

print("\nScaling features for ANN...")

# StandardScaler: Transforms features to have mean=0 and std=1
# Formula: X_scaled = (X - mean) / standard_deviation
# This is crucial for ANN as it helps with:
# 1. Faster convergence during training
# 2. Preventing features with larger ranges from dominating
# 3. Improved gradient descent optimization
# 4. Better numerical stability
scaler = StandardScaler()

# Fit scaler on training data and transform training features
# Fit: Calculate mean and std from training data
# Transform: Apply scaling using calculated statistics
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler (no fitting on test data to avoid data leakage)
# IMPORTANT: Use training statistics to scale test data
# This simulates real-world scenario where test data statistics are unknown
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# CALCULATE NEURAL NETWORK ARCHITECTURE PARAMETERS
# ============================================================================

# Determine number of neurons in hidden layers based on input features
# This is a heuristic approach; optimal sizes may vary by dataset
n_features = len(feature_cols)  # Number of input neurons

# Architecture 1: Single hidden layer with 2/3 of input features
# Rule of thumb: Hidden layer size between input and output size
neurons_single = int((2/3) * n_features)

# Architecture 2: Double hidden layers with decreasing sizes
# First layer: 2/3 of input features (broader representation)
# Second layer: 1/3 of input features (compressed representation)
# This creates a funnel architecture that progressively reduces dimensions
neurons_double_layer1 = int((2/3) * n_features)
neurons_double_layer2 = int((1/3) * n_features)

print(f"\nNumber of features: {n_features}")
print(f"Single layer neurons: {neurons_single}")
print(f"Double layer - Layer 1 neurons: {neurons_double_layer1}")
print(f"Double layer - Layer 2 neurons: {neurons_double_layer2}")

# ============================================================================
# DEFINE ANN ARCHITECTURES TO TEST
# ============================================================================

# List of different network architectures to compare
# Each architecture dictionary contains:
# - name: Identifier for file naming and referencing
# - hidden_layer_sizes: Tuple defining number of neurons in each hidden layer
# - description: Human-readable explanation of the architecture
architectures = [
    {
        'name': 'Single_Layer',
        'hidden_layer_sizes': (neurons_single,),  # Tuple with single value
        'description': f'Single hidden layer ({neurons_single} neurons)'
    },
    {
        'name': 'Double_Layer',
        'hidden_layer_sizes': (neurons_double_layer1, neurons_double_layer2),  # Tuple with two values
        'description': f'Double hidden layers ({neurons_double_layer1}, {neurons_double_layer2} neurons)'
    }
]

# ============================================================================
# SETUP CROSS-VALIDATION STRATEGY
# ============================================================================

# K-Fold Cross-Validation: Split training data into 10 folds
# Process: For each fold (k=1 to 10):
#   - Use 1 fold as validation set
#   - Use remaining 9 folds as training set
#   - Train model and make predictions on validation fold
# Result: Every sample gets predicted exactly once
# This provides robust performance estimates and helps detect overfitting
cv = KFold(n_splits=10,        # Number of folds
           shuffle=True,        # Randomly shuffle data before splitting
           random_state=42)     # Reproducible shuffling

# ============================================================================
# INITIALIZE RESULTS STORAGE
# ============================================================================

results = []  # List to store results from all architectures

print("\n" + "="*70)
print("Training ANN models with different architectures")
print("="*70)

# ============================================================================
# TRAIN AND EVALUATE MODELS FOR EACH ARCHITECTURE
# ============================================================================

for arch in architectures:
    print(f"\n{'='*70}")
    print(f"Training ANN with {arch['description']}")
    print(f"{'='*70}")
    
    # ========================================================================
    # INITIALIZE NEURAL NETWORK MODEL
    # ========================================================================
    
    # Multi-Layer Perceptron Regressor with specific hyperparameters
    # These parameters are carefully chosen for optimal performance
    ann_model = MLPRegressor(
        # Network architecture
        hidden_layer_sizes=arch['hidden_layer_sizes'],  # Number of neurons in each hidden layer
        
        # Activation function
        activation='relu',              # ReLU: max(0, x) - introduces non-linearity
                                       # Helps model learn complex patterns
                                       # Prevents vanishing gradient problem
        
        # Optimization algorithm
        solver='adam',                  # Adam: Adaptive Moment Estimation optimizer
                                       # Combines benefits of AdaGrad and RMSProp
                                       # Adaptive learning rates for each parameter
        
        # Regularization
        alpha=0.0001,                   # L2 penalty (regularization) parameter
                                       # Prevents overfitting by penalizing large weights
                                       # Formula: Loss = MSE + alpha * Σ(weights²)
        
        # Batch processing
        batch_size='auto',              # Automatically determine mini-batch size
                                       # Auto sets to min(200, n_samples)
        
        # Learning rate settings
        learning_rate='constant',       # Keep learning rate fixed during training
        learning_rate_init=0.001,       # Initial learning rate value
                                       # Controls step size in gradient descent
        
        # Training parameters
        max_iter=500,                   # Maximum number of training iterations
                                       # More iterations = more training time
        shuffle=True,                   # Shuffle samples in each iteration
                                       # Prevents learning order-dependent patterns
        random_state=42,                # Reproducible random weight initialization
                                       # Ensures consistent results across runs
        tol=0.0001,                     # Tolerance for optimization convergence
                                       # Training stops if improvement < tol for 2 consecutive epochs
        verbose=False,                  # Don't print training progress
                                       # Set to True for debugging
        
        # Advanced settings for SGD solver (used with Adam)
        warm_start=False,               # Don't reuse previous solution
                                       # Start fresh for each architecture
        momentum=0.9,                   # Momentum for gradient descent
                                       # Accelerates convergence, dampens oscillations
        nesterovs_momentum=True,        # Use Nesterov's momentum
                                       # Improved momentum variant with lookahead
        early_stopping=False,           # Don't stop training early
                                       # Set to True to use validation set for early stopping
        validation_fraction=0.1,        # Fraction of training data for validation
                                       # Only used if early_stopping=True
        
        # Adam optimizer parameters
        beta_1=0.9,                     # Exponential decay rate for 1st moment estimates
                                       # Controls memory of past gradients
        beta_2=0.999,                   # Exponential decay rate for 2nd moment estimates
                                       # Controls memory of past squared gradients
        epsilon=1e-08                   # Small constant for numerical stability
                                       # Prevents division by zero
    )
    
    # ========================================================================
    # TRAIN MODEL ON TRAINING SET
    # ========================================================================
    
    print("Training model...")
    # Fit the model to training data
    # This adjusts weights and biases through backpropagation
    ann_model.fit(X_train_scaled, y_train)  # Learn patterns from training data
    
    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================
    
    # Predictions on training set (to check for overfitting)
    # High training performance but low test performance indicates overfitting
    y_train_pred = ann_model.predict(X_train_scaled)
    
    # Predictions on test set (final model evaluation)
    # This represents real-world performance on unseen data
    y_test_pred = ann_model.predict(X_test_scaled)
    
    # Cross-validation predictions (to assess model generalization)
    print(f"Performing 10-fold cross-validation...")
    # cross_val_predict: Trains model on 9 folds, predicts on 1 fold, repeats 10 times
    # IMPORTANT: This uses out-of-fold predictions, not the trained model above
    # Each prediction is made by a model that hasn't seen that specific sample
    y_cv_pred = cross_val_predict(ann_model, X_train_scaled, y_train, 
                                   cv=cv, n_jobs=-1)  # n_jobs=-1: use all CPU cores for parallel processing
    
    # ========================================================================
    # CALCULATE PERFORMANCE METRICS
    # ========================================================================
    
    # --- Training Set Metrics ---
    # These show how well the model fits the training data
    
    # Pearson correlation: measures linear relationship (-1 to 1)
    # Formula: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
    # +1: Perfect positive correlation
    #  0: No linear correlation
    # -1: Perfect negative correlation
    train_corr, _ = pearsonr(y_train, y_train_pred)
    
    # R-squared: proportion of variance explained (can be negative to 1)
    # Formula: R² = 1 - (SS_res / SS_tot) where:
    # SS_res = Σ(yi - ŷi)² (residual sum of squares)
    # SS_tot = Σ(yi - ȳ)² (total sum of squares)
    # R² = 1: Perfect predictions
    # R² = 0: Model performs as well as mean baseline
    # R² < 0: Model performs worse than mean baseline
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Mean Absolute Error: average absolute difference (lower is better)
    # Formula: MAE = (1/n) × Σ|yi - ŷi|
    # Interpretable in the same units as the target variable
    # Less sensitive to outliers than MSE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # --- Cross-Validation Metrics ---
    # These metrics show how well the model generalizes to unseen data
    # More reliable than training metrics for assessing true performance
    cv_corr, _ = pearsonr(y_train, y_cv_pred)
    cv_r2 = r2_score(y_train, y_cv_pred)
    cv_mae = mean_absolute_error(y_train, y_cv_pred)
    
    # --- Test Set Metrics ---
    # Final evaluation on completely held-out data
    # This is the most honest assessment of model performance
    test_corr, _ = pearsonr(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # ========================================================================
    # STORE RESULTS FOR THIS ARCHITECTURE
    # ========================================================================
    
    # Store all results in a dictionary for later comparison and analysis
    results.append({
        'architecture': arch['name'],
        'description': arch['description'],
        'model': ann_model,
        # Training metrics
        'train_corr': train_corr,
        'train_r2': train_r2,
        'train_mae': train_mae,
        # Cross-validation metrics
        'cv_corr': cv_corr,
        'cv_r2': cv_r2,
        'cv_mae': cv_mae,
        # Test metrics
        'test_corr': test_corr,
        'test_r2': test_r2,
        'test_mae': test_mae,
        # Actual and predicted values for plotting
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_cv_pred': y_cv_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    })
    
    # ========================================================================
    # PRINT PERFORMANCE METRICS
    # ========================================================================
    
    print(f"\nTraining Set Metrics:")
    print(f"  Correlation Coefficient: {train_corr:.4f}")
    print(f"  R-squared: {train_r2:.4f}")
    print(f"  Mean Absolute Error: {train_mae:.4f}")
    
    print(f"\nCross-Validation Metrics (10-fold):")
    print(f"  Correlation Coefficient: {cv_corr:.4f}")
    print(f"  R-squared: {cv_r2:.4f}")
    print(f"  Mean Absolute Error: {cv_mae:.4f}")
    
    print(f"\nTest Set Metrics:")
    print(f"  Correlation Coefficient: {test_corr:.4f}")
    print(f"  R-squared: {test_r2:.4f}")
    print(f"  Mean Absolute Error: {test_mae:.4f}")

# ============================================================================
# CREATE SUMMARY TABLE OF ALL MODELS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY OF ALL MODELS")
print("="*70)

# Create DataFrame with all results for easy comparison
# This allows quick visual comparison of different architectures
summary_df = pd.DataFrame([
    {
        'Architecture': r['architecture'],
        'Description': r['description'],
        'Train_Corr': r['train_corr'],
        'Train_R2': r['train_r2'],
        'Train_MAE': r['train_mae'],
        'CV_Corr': r['cv_corr'],
        'CV_R2': r['cv_r2'],
        'CV_MAE': r['cv_mae'],
        'Test_Corr': r['test_corr'],
        'Test_R2': r['test_r2'],
        'Test_MAE': r['test_mae']
    }
    for r in results
])

# Display summary table in console with proper formatting
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# SAVE SUMMARY RESULTS TO CSV
# ============================================================================

summary_file = r'I:\JNK1-VERSION-7\JNK1-ANN-MODELING-RESULTS.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary saved to: {summary_file}")

# ============================================================================
# IDENTIFY AND SAVE BEST MODEL
# ============================================================================

print("\n" + "="*70)
print("Saving best model for future use...")
print("="*70)

# Find model with highest test R² (best predictive performance)
# Test R² is chosen as the selection criterion because:
# 1. It represents performance on completely unseen data
# 2. It's less prone to overfitting than training metrics
# 3. It's more interpretable than MAE for variance explanation
best_idx = summary_df['Test_R2'].idxmax()
best_arch = summary_df.loc[best_idx, 'Architecture']
best_desc = summary_df.loc[best_idx, 'Description']
best_result = results[best_idx]

print(f"\nBest model: {best_desc}")
print(f"Test R² = {best_result['test_r2']:.4f}")
print(f"Test MAE = {best_result['test_mae']:.4f}")

# ============================================================================
# SAVE TRAINED MODEL, SCALER, AND METADATA
# ============================================================================

# Save the trained neural network model using joblib
# joblib is preferred over pickle for large numpy arrays
best_model = best_result['model']
model_file = r'I:\JNK1-VERSION-7\JNK1-ANN-MODEL.pkl'
joblib.dump(best_model, model_file)
print(f"Model saved to: {model_file}")

# Save the feature scaler (needed for future predictions)
# CRITICAL: Must use the same scaler for new predictions
# New data must be scaled using training statistics
scaler_file = r'I:\JNK1-VERSION-7\JNK1-ANN-SCALER.pkl'
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to: {scaler_file}")

# Save feature information and model metadata
# This ensures reproducibility and proper use of the model
feature_info = {
    'feature_columns': feature_cols,          # List of feature names (order matters!)
    'n_features': len(feature_cols),          # Number of features
    'architecture': best_arch,                # Best architecture name
    'description': best_desc,                 # Architecture description
    'test_r2': best_result['test_r2'],        # Best model R²
    'test_mae': best_result['test_mae'],      # Best model MAE
    'authors': 'KR JINURAJ and V.N. BALAJI'  # Author information
}
feature_file = r'I:\JNK1-VERSION-7\JNK1-ANN-FEATURES.pkl'
joblib.dump(feature_info, feature_file)
print(f"Feature information saved to: {feature_file}")

# ============================================================================
# GENERATE VISUALIZATION PLOTS FOR EACH ARCHITECTURE
# ============================================================================

print("\n" + "="*70)
print("Generating plots...")
print("="*70)

for i, result in enumerate(results):
    arch_name = result['architecture']
    arch_desc = result['description']
    
    # ========================================================================
    # CREATE FIGURE WITH TWO SUBPLOTS
    # ========================================================================
    
    # Create figure with 2 side-by-side plots for comprehensive visualization
    # figsize=(12, 6): Width=12 inches, Height=6 inches (suitable for publication)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'ANN Regression ({arch_desc})\nTrue vs Predicted pIC50\nDeveloped by: KR JINURAJ and V.N. BALAJI', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust spacing between subplots for better layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.25)
    
    # ========================================================================
    # PLOT 1: TRAINING vs CROSS-VALIDATION
    # ========================================================================
    
    ax1 = axes[0]
    
    # Scatter plot: Training predictions (blue points)
    # Each point represents one molecule: x=true pIC50, y=predicted pIC50
    ax1.scatter(result['y_train'], result['y_train_pred'], 
                alpha=0.6,                  # Semi-transparent points to see overlapping
                label='Training',           # Legend label
                color='#2E86AB',           # Blue color (hex code)
                s=25,                       # Point size in points²
                edgecolors='none')          # No border around points
    
    # Scatter plot: Cross-validation predictions (purple points)
    # These are out-of-fold predictions, more reliable than training
    ax1.scatter(result['y_train'], result['y_cv_pred'], 
                alpha=0.6, 
                label='Cross-Validation (10-fold)', 
                color='#A23B72',           # Purple color
                s=25, 
                edgecolors='none')
    
    # Calculate trend line for training predictions
    # np.polyfit: fits a 1st degree polynomial (linear) to the data using least squares
    # Formula: minimizes Σ(yi - (mx + b))²
    # Returns [slope, intercept]
    z_train = np.polyfit(result['y_train'], result['y_train_pred'], 1)
    p_train = np.poly1d(z_train)  # Create polynomial function from coefficients
    x_trend_train = np.linspace(result['y_train'].min(), result['y_train'].max(), 100)
    ax1.plot(x_trend_train, p_train(x_trend_train), 
             color='#0066CC',           # Darker blue
             lw=2.5,                    # Line width
             label='Training Trend',    
             alpha=0.9, 
             linestyle='-')
    
    # Calculate trend line for cross-validation predictions
    z_cv = np.polyfit(result['y_train'], result['y_cv_pred'], 1)
    p_cv = np.poly1d(z_cv)
    x_trend_cv = np.linspace(result['y_train'].min(), result['y_train'].max(), 100)
    ax1.plot(x_trend_cv, p_cv(x_trend_cv), 
             color='#CC0066',           # Darker purple
             lw=2.5, 
             label='Validation Trend', 
             alpha=0.9, 
             linestyle='-')
    
    # Plot perfect prediction line (y = x)
    # This diagonal line shows where points would be if predictions were perfect
    # Points closer to this line indicate better predictions
    min_val = min(result['y_train'].min(), result['y_train_pred'].min(), 
                  result['y_cv_pred'].min())
    max_val = max(result['y_train'].max(), result['y_train_pred'].max(), 
                  result['y_cv_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'k--',                     # Black dashed line
             lw=2, 
             label='Perfect Prediction', 
             alpha=0.6, 
             linestyle='--')
    
    # Add text box with training trend equation
    # Shows the linear relationship: y = slope*x + intercept
    train_eq_text = f'Train: y = {z_train[0]:.3f}x + {z_train[1]:.3f}'
    ax1.text(0.05, 0.95,               # Position in axis coordinates (0-1)
             train_eq_text, 
             transform=ax1.transAxes,   # Use axis coordinates instead of data coordinates
             fontsize=10, 
             verticalalignment='top',   # Align to top of text
             color='#0066CC', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5',    # Rounded box with padding
                      facecolor='white',             # White background
                      alpha=0.85,                    # Semi-transparent
                      edgecolor='#0066CC',          # Blue border
                      linewidth=1.5))
    
    # Add text box with validation trend equation
    cv_eq_text = f'Val: y = {z_cv[0]:.3f}x + {z_cv[1]:.3f}'
    ax1.text(0.05, 0.85, 
             cv_eq_text, 
             transform=ax1.transAxes, 
             fontsize=10, 
             verticalalignment='top', 
             color='#CC0066', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      alpha=0.85, 
                      edgecolor='#CC0066', 
                      linewidth=1.5))
    
    # Set axis labels with proper formatting
    ax1.set_xlabel('True pIC50', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted pIC50', fontsize=13, fontweight='bold')
    
    # Set plot title with performance metrics
    # Shows key statistics for quick assessment
    ax1.set_title(f'Training vs Cross-Validation\n'
                  f'Train: R²={result["train_r2"]:.3f}, MAE={result["train_mae"]:.3f}\n'
                  f'CV: R²={result["cv_r2"]:.3f}, MAE={result["cv_mae"]:.3f}',
                  fontsize=12, fontweight='bold', pad=10)
    
    # Set equal aspect ratio (square plot)
    # Ensures 1 unit on x-axis = 1 unit on y-axis visually
    ax1.set_aspect('equal', adjustable='box')
    
    # Add legend in lower right corner
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='black')
    
    # Add grid for easier reading of values
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Adjust tick label size for readability
    ax1.tick_params(labelsize=11)
    
    # ========================================================================
    # PLOT 2: TRAINING vs TEST SET
    # ========================================================================
    
    ax2 = axes[1]
    
    # Scatter plot: Training predictions (same as plot 1)
    ax2.scatter(result['y_train'], result['y_train_pred'], 
                alpha=0.6, 
                label='Training', 
                color='#2E86AB',           # Blue
                s=25, 
                edgecolors='none')
    
    # Scatter plot: Test predictions (green points)
    # Most important metric - shows performance on completely unseen data
    ax2.scatter(result['y_test'], result['y_test_pred'], 
                alpha=0.6, 
                label='Test', 
                color='#06A77D',           # Green color
                s=25, 
                edgecolors='none')
    
    # Training trend line (same calculation as plot 1)
    z_train2 = np.polyfit(result['y_train'], result['y_train_pred'], 1)
    p_train2 = np.poly1d(z_train2)
    x_trend_train2 = np.linspace(result['y_train'].min(), result['y_train'].max(), 100)
    ax2.plot(x_trend_train2, p_train2(x_trend_train2), 
             color='#0066CC', 
             lw=2.5, 
             label='Training Trend', 
             alpha=0.9, 
             linestyle='-')
    
    # Test trend line
    # If significantly different from training trend, may indicate overfitting
    z_test = np.polyfit(result['y_test'], result['y_test_pred'], 1)
    p_test = np.poly1d(z_test)
    x_trend_test = np.linspace(result['y_test'].min(), result['y_test'].max(), 100)
    ax2.plot(x_trend_test, p_test(x_trend_test), 
             color='#00AA66',           # Darker green
             lw=2.5, 
             label='Test Trend', 
             alpha=0.9, 
             linestyle='-')
    
    # Perfect prediction line (y = x)
    min_val = min(result['y_train'].min(), result['y_train_pred'].min(),
                  result['y_test'].min(), result['y_test_pred'].min())
    max_val = max(result['y_train'].max(), result['y_train_pred'].max(),
                  result['y_test'].max(), result['y_test_pred'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 
             'k--', 
             lw=2, 
             label='Perfect Prediction', 
             alpha=0.6, 
             linestyle='--')
    
    # Training trend equation text box
    train_eq_text2 = f'Train: y = {z_train2[0]:.3f}x + {z_train2[1]:.3f}'
    ax2.text(0.05, 0.95, 
             train_eq_text2, 
             transform=ax2.transAxes, 
             fontsize=10, 
             verticalalignment='top', 
             color='#0066CC', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      alpha=0.85, 
                      edgecolor='#0066CC', 
                      linewidth=1.5))
    
    # Test trend equation text box
    test_eq_text = f'Test: y = {z_test[0]:.3f}x + {z_test[1]:.3f}'
    ax2.text(0.05, 0.85, 
             test_eq_text, 
             transform=ax2.transAxes, 
             fontsize=10, 
             verticalalignment='top', 
             color='#00AA66', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      alpha=0.85, 
                      edgecolor='#00AA66', 
                      linewidth=1.5))
    
    # Set axis labels
    ax2.set_xlabel('True pIC50', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Predicted pIC50', fontsize=13, fontweight='bold')
    
    # Set plot title with performance metrics
    ax2.set_title(f'Training vs Test Set\n'
                  f'Train: R²={result["train_r2"]:.3f}, MAE={result["train_mae"]:.3f}\n'
                  f'Test: R²={result["test_r2"]:.3f}, MAE={result["test_mae"]:.3f}',
                  fontsize=12, fontweight='bold', pad=10)
    
    # Set equal aspect ratio
    ax2.set_aspect('equal', adjustable='box')
    
    # Add legend
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='black')
    
    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Adjust tick labels
    ax2.tick_params(labelsize=11)
    
    # ========================================================================
    # SAVE PLOT TO FILE
    # ========================================================================
    
    # Adjust layout to prevent overlapping elements
    # rect: [left, bottom, right, top] in figure coordinates
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save plot as high-resolution PNG image
    # dpi=300: Publication quality (print-ready)
    # bbox_inches='tight': Remove extra whitespace
    plot_file = r'I:\JNK1-VERSION-7\JNK1-ANN-MODELING-PLOT-{}.png'.format(arch_name.lower())
    plt.savefig(plot_file, 
                dpi=300,               # High resolution (300 dots per inch)
                bbox_inches='tight')   # Tight bounding box (no extra whitespace)
    
    print(f"Plot saved: {plot_file}")
    
    # Close figure to free memory
    # Important when generating multiple plots
    plt.close()

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("ANN Modeling complete! All results and plots have been saved.")
print("Developed by: KR JINURAJ and V.N. BALAJI")
print("="*70)

# ============================================================================
# VERIFY ALL OUTPUT FILES WERE CREATED SUCCESSFULLY
# ============================================================================

print("\n" + "="*70)
print("VERIFICATION: Checking all output files...")
print("="*70)

# Define expected output files with descriptions
# This verification ensures all outputs were generated correctly
output_files = {
    'Summary Results (CSV)': summary_file,
    'Trained Model': model_file,
    'Feature Scaler': scaler_file,
    'Feature Information': feature_file,
    'Single Layer Plot': r'I:\JNK1-VERSION-7\JNK1-ANN-MODELING-PLOT-single_layer.png',
    'Double Layer Plot': r'I:\JNK1-VERSION-7\JNK1-ANN-MODELING-PLOT-double_layer.png'
}

# Check each file exists and report its size
all_files_ok = True
for file_description, file_path in output_files.items():
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        # Format file size with thousand separators for readability
        print(f"✓ {file_description}: OK ({file_size:,} bytes)")
    else:
        print(f"✗ {file_description}: MISSING!")
        all_files_ok = False

# ============================================================================
# DISPLAY FINAL STATUS
# ============================================================================

print("\n" + "="*70)
if all_files_ok:
    print("SUCCESS: All files created successfully!")
    print("="*70)
    print("\nModel Performance Summary:")
    print(f"  Best Architecture: {best_arch}")
    print(f"  Test R²: {best_result['test_r2']:.4f}")
    print(f"  Test MAE: {best_result['test_mae']:.4f}")
    print(f"  Test Correlation: {best_result['test_corr']:.4f}")
    print("\nOutput Location: I:\\JNK1-VERSION-7\\")
else:
    print("WARNING: Some files were not created!")
    print("="*70)

print("\n" + "="*70)
print("Script execution completed.")
print("="*70)

# Pause and wait for user input before closing
# This prevents the console window from closing immediately
# Allows user to review results before exiting
input("\nPress Enter to exit...")
