#include <stdlib.h>
#include <filesystem>
#include <libxml/tree.h>
#include <libxml/xpath.h>

namespace fs = std::filesystem;

class XMLPatcher
{
    fs::path xml_file;
    xmlDocPtr doc;
    xmlXPathContextPtr xpathCtx;
    
  public: 
    XMLPatcher(fs::path _xml_file);
    ~XMLPatcher();
    void changeXML(std::string xpath, std::string new_xml);
    void removeXML(std::string xpath);
    void addXML(std::string xpath, std::string new_xml);
    std::string getXML();
};