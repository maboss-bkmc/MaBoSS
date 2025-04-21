#ifndef BN_EXCEPTION_H
#define BN_EXCEPTION_H

#include <string>
#include <iostream>
class BNException {

    std::string msg;
  
  public:
    BNException(const std::string& _msg) : msg(_msg) { }
  
    const std::string& getMessage() const {return msg;}
};

std::ostream& operator<<(std::ostream& os, const BNException& e);
#endif