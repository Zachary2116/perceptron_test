#include <stdio.h>

int dot_product(const int *a, const int *b, size_t length) {
    int result = 0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

