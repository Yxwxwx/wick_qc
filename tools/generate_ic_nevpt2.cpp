#include "method/ic_nevpt2.h"

#include <iostream>

int main() {
  std::cout << wickqc::method::IcNevpt2Generator().GenerateNumpy();
}
