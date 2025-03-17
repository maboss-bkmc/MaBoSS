#ifndef BN_EXCEPTION_H
#define BN_EXCEPTION_H

#include <string>

class BNException {

    std::string msg;
  
  public:
    BNException(const std::string& _msg) : msg(_msg) { }
  
    const std::string& getMessage() const {return msg;}
};

#endif