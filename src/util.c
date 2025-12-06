#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    size_t n_features;   // number of input features
    int   *weights;      // array of length n_features
    int    bias;         // bias term
    int    learning_rate;
} Parameters;

int dot_product(const int *a, const int *b, size_t length) {
    int result = 0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int weighted_sum(int x1, int x2, int w1, int w2) {
    int values[2]  = { x1, x2 };
    int weights[2] = { w1, w2 };
    return dot_product(values, weights, 2);
}

int insert_into_array(int *array, size_t length, int value) {
    for (size_t i = 0; i < length; i++) {
        if (array[i] == 0) {   // 0 means "empty slot"
            array[i] = value;
            return 1;          // success
        }
    }
    return 0;                  // no empty slot found
}

int sorted_insert(int *yes, int *no, int value) {
    if (value % 2 == 0) {
        return insert_into_array(yes, 10, value);  // even -> yes
    } else {
        return insert_into_array(no, 10, value);   // odd  -> no
    }
}

int count_columns(FILE *file) {
    int count = 0;
    int value;
    int c;

    // Read ONLY the first line: count how many ints are on it
    while (fscanf(file, "%d", &value) == 1) {
        count++;

        // Look at the next character after the int
        c = fgetc(file);

        // End of line or file → we’re done counting columns
        if (c == '\n' || c == EOF) {
            break;
        }
        // Otherwise it's likely space, just keep going
    }

    rewind(file);  // reset file position for later reads
    return count;
}

int count_rows(FILE *file) {
    int count = 0;
    int c;
    int last_char = '\n'; // pretend we just saw a newline before the file

    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            count++;
        }
        last_char = c;
    }

    // If file doesn't end with '\n' but has content, count the last line
    if (last_char != '\n') {
        count++;
    }

    rewind(file); // reset file position for later reads
    return count;
}
