#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

double dtw_distance(double *seq1, double *seq2, int len1, int len2, int dim)
{
    int i, j, m;
    double cost, min_cost;

    double **dtw_matrix = (double **)malloc((len1 + 1) * sizeof(double *));
    for (i = 0; i <= len1; i++)
    {
        dtw_matrix[i] = (double *)malloc((len2 + 1) * sizeof(double));
        for (j = 0; j <= len2; j++)
        {
            dtw_matrix[i][j] = INFINITY;
        }
    }

    dtw_matrix[0][0] = 0.0;

    for (i = 1; i <= len1; i++)
    {
        for (j = 1; j <= len2; j++)
        {
            cost = 0.0;
            for (m = 0; m < dim; m++)
            {
                double diff = seq1[(i - 1) * dim + m] - seq2[(j - 1) * dim + m];
                cost += diff * diff;
            }

            min_cost = fmin(dtw_matrix[i - 1][j], fmin(dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]));
            dtw_matrix[i][j] = cost + min_cost;
        }
    }

    double result = dtw_matrix[len1][len2];

    for (i = 0; i <= len1; i++)
    {
        free(dtw_matrix[i]);
    }
    free(dtw_matrix);

    return result;
}

void squared_euclidean_distance(double *X, double *Y, double *D, int rows_X, int cols, int rows_Y)
{
    for (int i = 0; i < rows_X; i++)
    {
        for (int j = 0; j < rows_Y; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; k++)
            {
                double diff = X[i * cols + k] - Y[j * cols + k];
                sum += diff * diff;
            }
            D[i * rows_Y + j] = sum;
        }
    }
}

void rbf_kernel(double *D, double *K, int rows_X, int rows_Y, double sigma)
{
    double factor = -1.0 / (2.0 * sigma * sigma);
    for (int i = 0; i < rows_X * rows_Y; i++)
    {
        K[i] = exp(D[i] * factor);
    }
}

double compute_mmd(double *X, double *Y, int rows_X, int rows_Y, int cols, double sigma)
{
    double *D_xx = (double *)malloc(rows_X * rows_X * sizeof(double));
    double *D_yy = (double *)malloc(rows_Y * rows_Y * sizeof(double));
    double *D_xy = (double *)malloc(rows_X * rows_Y * sizeof(double));
    double *K_xx = (double *)malloc(rows_X * rows_X * sizeof(double));
    double *K_yy = (double *)malloc(rows_Y * rows_Y * sizeof(double));
    double *K_xy = (double *)malloc(rows_X * rows_Y * sizeof(double));

    // Compute distance matrices
    squared_euclidean_distance(X, X, D_xx, rows_X, cols, rows_X);
    squared_euclidean_distance(Y, Y, D_yy, rows_Y, cols, rows_Y);
    squared_euclidean_distance(X, Y, D_xy, rows_X, cols, rows_Y);

    // Compute RBF kernel matrices
    rbf_kernel(D_xx, K_xx, rows_X, rows_X, sigma);
    rbf_kernel(D_yy, K_yy, rows_Y, rows_Y, sigma);
    rbf_kernel(D_xy, K_xy, rows_X, rows_Y, sigma);

    // Compute MMD score
    double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0;
    for (int i = 0; i < rows_X * rows_X; i++)
        sum_xx += K_xx[i];
    for (int i = 0; i < rows_Y * rows_Y; i++)
        sum_yy += K_yy[i];
    for (int i = 0; i < rows_X * rows_Y; i++)
        sum_xy += K_xy[i];

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