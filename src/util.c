#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    size_t n_features;   // number of inputs
    double *weights;     // weight array
    double  bias;        // bias term
    double  learning_rate;
} Parameters;

double dot_product(const double *a, const double *b, size_t length) {
    double result = 0.0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

double weighted_sum(double x1, double x2, double w1, double w2) {
    double values[2]  = { x1, x2 };
    double weights[2] = { w1, w2 };
    return dot_product(values, weights, 2);
}

// Count how many numbers are on the first line
int count_columns(FILE *file) {
    int count = 0;
    double value;
    int c;

    while (fscanf(file, "%lf", &value) == 1) {
        count++;

        c = fgetc(file);
        if (c == '\n' || c == EOF) {
            break;
        }
    }

    rewind(file);
    return count;
}

// Count how many lines are in the file
int count_rows(FILE *file) {
    int count = 0;
    int c;
    int last_char = '\n';

    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            count++;
        }
        last_char = c;
    }

    if (last_char != '\n') {
        count++;
    }

    rewind(file);
    return count;
}
