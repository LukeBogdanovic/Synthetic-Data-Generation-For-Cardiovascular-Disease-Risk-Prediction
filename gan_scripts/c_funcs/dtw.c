/**
 * @file dtw.c
 * @author Luke Bogdanovic
 * @date 16/02/2025
 * @brief Implementation of similarity and distance metrics for multivariate time series data.
 *        Functions include:
 *          - dtw_distance
 *          - squared_euclidean_distance
 *          - rbf_kernel
 *          - compute_mmd
 *        This file is used from a python file to calculate the MVDTW and MMD metrics during
 *        the generation of electrocardiogram signals.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

/**
 * @brief Computes the dynamic time warping distance between two multivariate time series.
 *
 * Aligns sequences by warping the time axis to minimize the cumulative distance for multivariate
 * time series inputs. Gives a measure of the similarity between multi lead ECGs from a real and
 * generated sample by calculating their distances using Euclidean distance.
 *
 * @param seq1 First flattened sequence
 * @param seq2 Second flattened sequence
 * @param len1 Number of time steps in the first sequence
 * @param len2 Number of time steps in the second sequence
 * @param dim Number of dimensions/leads per time step
 * @return dtw distance between the two sequences
 */
double dtw_distance(double *seq1, double *seq2, int len1, int len2, int dim)
{
    int i, j, m;                                                            // Declare variables for loops
    double cost, min_cost;                                                  // Declare cost and min_cost variables
    double **dtw_matrix = (double **)malloc((len1 + 1) * sizeof(double *)); // Allocate memory for the dtw cost matrix rows
    for (i = 0; i <= len1; i++)
    {
        dtw_matrix[i] = (double *)malloc((len2 + 1) * sizeof(double)); // Allocate memory for the dtw cost matrix cols per column
        for (j = 0; j <= len2; j++)
        {
            dtw_matrix[i][j] = INFINITY; // Initialise all matrix cells to infinity
        }
    }
    dtw_matrix[0][0] = 0.0; // Set initial value to 0
    for (i = 1; i <= len1; i++)
    {
        for (j = 1; j <= len2; j++)
        {
            cost = 0.0; // Initialise cost sum variable
            // Compute the squared Euclidean distance between the i-th and j-th time step (across all dimensions)
            for (m = 0; m < dim; m++)
            {
                double diff = seq1[(i - 1) * dim + m] - seq2[(j - 1) * dim + m]; // Get the differebce for the dimension/lead m
                cost += diff * diff;                                             // Add squared difference to the sum
            }
            min_cost = fmin(dtw_matrix[i - 1][j], fmin(dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])); // Get the minimum of three adjacent paths
            dtw_matrix[i][j] = cost + min_cost;
        }
    }
    double result = dtw_matrix[len1][len2]; // Get the final dtw distance measurement from the bottom right point of the matrix
    // Free all allocated memory
    for (i = 0; i <= len1; i++)
    {
        free(dtw_matrix[i]);
    }
    free(dtw_matrix);
    return result; // Return final distance measurement
}

/**
 * @brief Computes the pairwise squared Euclidean distances between rows in X and Y.
 *
 * Calculates how different each sample in X is from each sample in Y.
 * Computes the squared distance between sample from X and Y by comparing all features.
 *
 * @param X Flattened 2D array for X
 * @param Y Flattened 2D array for Y
 * @param D Output matrix, rows_X*rows_Y in size
 * @param rows_X Number of samples in dataset X
 * @param cols Number of features per sample
 * @param rows_Y Number of samples in dataset Y
 */
void squared_euclidean_distance(double *X, double *Y, double *D, int rows_X, int cols, int rows_Y)
{
    for (int i = 0; i < rows_X; i++)
    {
        for (int j = 0; j < rows_Y; j++)
        {
            double sum = 0.0;              // Initialise sum for squared differences
            for (int k = 0; k < cols; k++) // Loop for each feature
            {
                double diff = X[i * cols + k] - Y[j * cols + k]; // Compute the difference for the feature
                sum += diff * diff;                              // Add squared difference to the sum
            }
            D[i * rows_Y + j] = sum; // Store the distance in the flattened matrix
        }
    }
}

/**
 * @brief Applies the RBF kernel to a squared distance matrix.
 *
 * @param D Input flattened distance matrix
 * @param K Outut flattened kernel matrix
 * @param rows_X Number of samples in dataset X
 * @param rows_Y Number of samples in dataset Y
 * @param sigma Bandwidth parameter for the RBF kernel
 */
void rbf_kernel(double *D, double *K, int rows_X, int rows_Y, double sigma)
{
    double factor = -1.0 / (2.0 * sigma * sigma); // Compute the scaling factor for exponent
    for (int i = 0; i < rows_X * rows_Y; i++)     // Loop for all matrix entries
    {
        K[i] = exp(D[i] * factor); // Apply RBF kernel formula: e^(-d)
    }
}

/**
 * @brief Computes the Maximum Mean Discrepancy (MMD) between two datasets using an RBF kernel.
 * MMD is a similarity measurement used between two distributions, in this case a real and
 * a synthetic set of data. A lower score indicates similar distributions.
 *
 * @param X Flattened real dataset (rows_X, cols)
 * @param Y Flattened synthetic dataset (rows_Y, cols)
 * @param rows_X Number of samples in X
 * @param rows_Y Number of samples in Y
 * @param cols Number of features per sample
 * @param sigma Bandwidth for the RBF kernel
 * @return MMD score
 */
double compute_mmd(double *X, double *Y, int rows_X, int rows_Y, int cols, double sigma)
{
    // Allocate memory for each of the distance matrices
    double *D_xx = (double *)malloc(rows_X * rows_X * sizeof(double)); // Allocation for real-real
    double *D_yy = (double *)malloc(rows_Y * rows_Y * sizeof(double)); // Allocation for fake-fake
    double *D_xy = (double *)malloc(rows_X * rows_Y * sizeof(double)); // Allocation for real-fake
    // Allocate memory for each of the kernel matrices
    double *K_xx = (double *)malloc(rows_X * rows_X * sizeof(double)); // Allocation for real-real
    double *K_yy = (double *)malloc(rows_Y * rows_Y * sizeof(double)); // Allocation for fake-fake
    double *K_xy = (double *)malloc(rows_X * rows_Y * sizeof(double)); // Allocation for real-fake
    // Compute pairwise squared distances between
    squared_euclidean_distance(X, X, D_xx, rows_X, cols, rows_X); // Compute squared distance between the real and real dataset
    squared_euclidean_distance(Y, Y, D_yy, rows_Y, cols, rows_Y); // Compute squared distance between the fake and fake dataset
    squared_euclidean_distance(X, Y, D_xy, rows_X, cols, rows_Y); // Compute squared distance between the real and fake dataset
    // Apply the RBF kernel to the distance matrices
    rbf_kernel(D_xx, K_xx, rows_X, rows_X, sigma);   // Kernel for real-real applied to distance matrix for real-real
    rbf_kernel(D_yy, K_yy, rows_Y, rows_Y, sigma);   // Kernel for fake-fake applied to distance matrix for fake-fake
    rbf_kernel(D_xy, K_xy, rows_X, rows_Y, sigma);   // Kernel for real-fake applied to distance matrix for real-fake
    double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0; // Initialise kernel matrix sum values
    for (int i = 0; i < rows_X * rows_X; i++)
        sum_xx += K_xx[i]; // Sum all elements in K_xx
    for (int i = 0; i < rows_Y * rows_Y; i++)
        sum_yy += K_yy[i]; // Sum all elements in K_yy
    for (int i = 0; i < rows_X * rows_Y; i++)
        sum_xy += K_xy[i]; // Sum all elements in K_xy
    // Compute the MMD score using the MMD formula: MMD = mean(K_xx) + mean(K_yy) - 2*mean(K_xy)
    double mmd_score = (sum_xx / (rows_X * rows_X)) +
                       (sum_yy / (rows_Y * rows_Y)) -
                       (2.0 * sum_xy / (rows_X * rows_Y));
    // Free allocated memory
    free(D_xx);
    free(D_yy);
    free(D_xy);
    free(K_xx);
    free(K_yy);
    free(K_xy);
    return mmd_score;
}