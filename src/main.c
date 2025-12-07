// main.c
#include <stdio.h>
#include <stdlib.h>
#include "training.c"   // brings in util.c, Parameters, train(), dot_product(), count_*()

int main(void) {
    const char *train_filename = "../data_folder/training_data.txt";
    const char *test_filename  = "../data_folder/input_data.txt";
    const char *pred_filename  = "../data_folder/predictions.txt";

    /* ====== TRAINING DATA ====== */

    FILE *train_file = fopen(train_filename, "r");
    if (!train_file) {
        perror("Error opening training file");
        return 1;
    }

    int n_cols_train = count_columns(train_file);  // features + label
    int n_rows_train = count_rows(train_file);     // number of training rows

    if (n_cols_train <= 1 || n_rows_train <= 0) {
        fprintf(stderr, "Invalid training data format\n");
        fclose(train_file);
        return 1;
    }

    int n_features = n_cols_train - 1;             // last column is label

    double *train_inputs  = malloc(n_rows_train * n_features * sizeof(double));
    int    *train_outputs = malloc(n_rows_train * sizeof(int));
    if (!train_inputs || !train_outputs) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        fclose(train_file);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    // read training data
    for (int i = 0; i < n_rows_train; i++) {
        for (int j = 0; j < n_cols_train; j++) {
            double value;
            if (fscanf(train_file, "%lf", &value) != 1) {
                fprintf(stderr, "Error reading training data at row %d col %d\n", i, j);
                fclose(train_file);
                free(train_inputs);
                free(train_outputs);
                return 1;
            }

            if (j < n_features) {
                train_inputs[i * n_features + j] = value;
            } else {
                train_outputs[i] = (int)value;     // assume 0 or 1
            }
        }
    }
    fclose(train_file);

    /* ====== SET UP AND TRAIN PERCEPTRON ====== */

    Parameters params;
    params.n_features    = (size_t)n_features;
    params.learning_rate = 0.1;
    params.bias          = 0.0;

    double *weights = calloc(n_features, sizeof(double));
    if (!weights) {
        fprintf(stderr, "Memory allocation failed for weights\n");
        free(train_inputs);
        free(train_outputs);
        return 1;
    }
    params.weights = weights;

    printf("Training on %d samples with %d features...\n", n_rows_train, n_features);
    train(&params, train_inputs, train_outputs, (size_t)n_rows_train);

    printf("Trained weights: ");
    for (size_t i = 0; i < params.n_features; i++) {
        printf("%f ", params.weights[i]);
    }
    printf("\nTrained bias: %f\n", params.bias);

    /* ====== TEST DATA (PREDICTIONS) ====== */

    FILE *test_file = fopen(test_filename, "r");
    if (!test_file) {
        perror("Error opening test file");
        free(weights);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    int n_cols_test = count_columns(test_file);   // should be == n_features
    int n_rows_test = count_rows(test_file);

    if (n_cols_test != n_features) {
        fprintf(stderr,
                "Test data has %d columns, but model expects %d features\n",
                n_cols_test, n_features);
        fclose(test_file);
        free(weights);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    if (n_rows_test <= 0) {
        fprintf(stderr, "No test data found\n");
        fclose(test_file);
        free(weights);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    double *test_inputs = malloc(n_rows_test * n_features * sizeof(double));
    if (!test_inputs) {
        fprintf(stderr, "Memory allocation failed for test data\n");
        fclose(test_file);
        free(weights);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    // read test data (features only, no labels)
    for (int i = 0; i < n_rows_test; i++) {
        for (int j = 0; j < n_features; j++) {
            double value;
            if (fscanf(test_file, "%lf", &value) != 1) {
                fprintf(stderr, "Error reading test data at row %d col %d\n", i, j);
                fclose(test_file);
                free(test_inputs);
                free(weights);
                free(train_inputs);
                free(train_outputs);
                return 1;
            }
            test_inputs[i * n_features + j] = value;
        }
    }
    fclose(test_file);

    /* ====== WRITE PREDICTIONS TO FILE ====== */

    FILE *pred_file = fopen(pred_filename, "w");
    if (!pred_file) {
        perror("Error opening predictions file for writing");
        free(test_inputs);
        free(weights);
        free(train_inputs);
        free(train_outputs);
        return 1;
    }

    for (int i = 0; i < n_rows_test; i++) {
        const double *x = &test_inputs[i * n_features];
        double activation = dot_product(params.weights, x, params.n_features)
                            + params.bias;
        int prediction = (activation >= 0.0) ? 1 : 0;

        // each line: just the predicted label
        fprintf(pred_file, "%d\n", prediction);
    }

    fclose(pred_file);

    printf("Wrote %d predictions to %s\n", n_rows_test, pred_filename);

    /* ====== CLEANUP ====== */

    free(test_inputs);
    free(weights);
    free(train_inputs);
    free(train_outputs);

    return 0;
}
