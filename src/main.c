#include "util.c"
#include <stdio.h>

int main() {
    int vec1[] = {1, 2, 3};
    int vec2[] = {4, 5, 6};
    size_t length = 3;

    int result = dot_product(vec1, vec2, length);
    printf("Dot product: %d\n", result);

    return 0;
}

