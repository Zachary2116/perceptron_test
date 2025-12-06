// main.c
#include <stdio.h>
#include <stdlib.h>
#include "training.c"   // pulls in util.c and Parameters, train(), etc.

int main(void) {
    const char *filename = "../data_folder/input_data.txt";

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening input_data.txt");
        return 1;
    }

    // Use helpers from util.c
    int n_cols = count_columns(file);  // total columns (features + label)
    int n_rows = count_rows(file);    // number of samples

    if (n_cols <= 1 || n_rows <= 0) {
        fprintf(stderr, "Invalid data format\n");
        fclose(file);
        return 1;
    }

    int n_features = n_cols - 1;  // last column is the label

    // Allocate arrays for inputs and outputs
    int *inputs  = malloc(n_rows * n_features * sizeof(int));
    int *outputs = malloc(n_rows * sizeof(int));
    if (!inputs || !outputs) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        free(inputs);
        free(outputs);
        return 1;
    }

    // Read data from file
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int value;
            if (fscanf(file, "%d", &value) != 1) {
                fprintf(stderr, "Error reading data at row %d col %d\n", i, j);
                fclose(file);
                free(inputs);
                free(outputs);
                return 1;
            }

            if (j < n_features) {
                inputs[i * n_features + j] = value;  // feature
            } else {
                outputs[i] = value;                  // label
            }
        }
    }
    fclose(file);

    // Set up perceptron parameters
    Parameters params;
    params.n_features    = (size_t)n_features;
    params.learning_rate = 1;
    params.bias          = 0;

    int *weights = calloc(n_features, sizeof(int)); // start from zeros
    if (!weights) {
        fprintf(stderr, "Memory allocation for weights failed\n");
        free(inputs);
        free(outputs);
        return 1;
    }
    params.weights = weights;

    // Train perceptron
    train(&params, inputs, outputs, (size_t)n_rows);

    // Print trained weights and bias
    printf("Trained weights: ");
    for (size_t i = 0; i < params.n_features; i++) {
        printf("%d ", params.weights[i]);
    }
    printf("\nTrained bias: %d\n", params.bias);

    // Clean up
    free(weights);
    free(inputs);
    free(outputs);

    return 0;
}
