#include<iostream>
#include<cmath>

double manySqrt(double x){
    /* This loop should run roughly 1e111 iterations. At 3GHz, this is
    10^92 times the age of the universe */
    for (double z = 0; z < 1e100; z += 1e-10){ 
        sqrt(x);
    }
    return 1.0;
}

int main(void){
    for (int i = 0; i < 100; i++){
        manySqrt(2.0);
    }
    std::cout << "Done!" << std::endl;
}
