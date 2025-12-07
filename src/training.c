#include <stdio.h>
#include "util.c"

#define EPOCHS 1000

// One update step on a single sample
void train_perceptron(Parameters *params,
                      const double *inputs,
                      int expected_output)
{
    double activation = dot_product(params->weights,
                                    inputs,
                                    params->n_features)
                        + params->bias;

    int prediction = (activation >= 0.0) ? 1 : 0;
    int error      = expected_output - prediction;

    if (error == 0) return;

    for (size_t i = 0; i < params->n_features; i++) {
        params->weights[i] += params->learning_rate * error * inputs[i];
    }

    params->bias += params->learning_rate * error;
}

// Run training over all samples for a few epochs
void train(Parameters *params,
           const double *inputs,
           const int *outputs,
           size_t data_size)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t i = 0; i < data_size; i++) {
            const double *sample_inputs = &inputs[i * params->n_features];
            int expected_output         = outputs[i];
            train_perceptron(params, sample_inputs, expected_output);
        }
    }
}
