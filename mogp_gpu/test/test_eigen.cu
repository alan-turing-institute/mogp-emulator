#include <iostream>

#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <stdexcept>
#include <random>

#include "../src/types.hpp"
#include "../src/util.hpp"

typedef double REAL;


void test_dot_product_vv() {
    vec x1(4);
    x1 << 1., 4., 3., 6.;
    vec x2(4);
    x2 << 1., 2, 3., 5.;
    auto result = x1.dot(x2);
    std::cout<<" result vv "<< result <<std::endl;
 
}

void test_dot_product_mv() {
    mat x1(3,3);
    x1 << 1., 4., 3.,
          2., 3., 4.,
          3., 4., 5.;
    vec x2(3);
    x2 << 1., 2, 3.;
    auto result = x1 * x2;
    std::cout<<" result mv "<< result <<std::endl;
 
}

int main(void)
{
    test_dot_product_vv(); 
    test_dot_product_mv();
    return 0;
}
