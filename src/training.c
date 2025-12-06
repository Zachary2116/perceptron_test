// training.c
#include <stdio.h>
#include "util.c"   // must define Parameters and dot_product

#define EPOCHS 40

// Train once on a single sample (n parameters)
void train_perceptron(Parameters *params,
                      const int *inputs,      // length = params->n_features
                      int expected_output)    // 0 or 1
{
    // activation = w·x + b
    int activation = dot_product(params->weights,
                                 inputs,
                                 params->n_features)
                     + params->bias;

    int prediction = (activation >= 0) ? 1 : 0;
    int error      = expected_output - prediction;   // -1, 0, or 1

    if (error == 0) return; // already correct

    // update each weight with its matching input
    for (size_t i = 0; i < params->n_features; i++) {
        params->weights[i] += params->learning_rate * error * inputs[i];
    }

    // update bias
    params->bias += params->learning_rate * error;
}

// Train over a dataset with n parameters
// inputs is a flat array: sample 0 is at indices [0 .. n_features-1],
// sample 1 at [n_features .. 2*n_features-1], etc.
void train(Parameters *params,
           const int *inputs,     // length = data_size * params->n_features
           const int *outputs,    // length = data_size
           size_t data_size)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t i = 0; i < data_size; i++) {
            const int *sample_inputs = &inputs[i * params->n_features];
            int expected_output      = outputs[i];
            train_perceptron(params, sample_inputs, expected_output);
        }
    }
}
