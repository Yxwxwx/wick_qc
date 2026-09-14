#include "method/ccsd.h"

#include <iostream>

int main() {
  std::cout << wickqc::method::CcsdGenerator().GenerateNumpy();
}
