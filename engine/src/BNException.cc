#include "BNException.h"

std::ostream& operator<<(std::ostream& os, const BNException& e)
{
  os << "BooleanNetwork exception: " << e.getMessage() << '\n';
  return os;
}
