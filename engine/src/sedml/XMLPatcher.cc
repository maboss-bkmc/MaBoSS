#include "XMLPatcher.h"

#include <sbml/xml/XMLError.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include "SedException.h"

XMLPatcher::XMLPatcher(fs::path _xml_file): xml_file(_xml_file)
{
    /* Init libxml */     
    xmlInitParser();
    
    /* Load XML document */
    doc = xmlParseFile(xml_file.c_str());
    /* Create xpath evaluation context */
    xpathCtx = xmlXPathNewContext(doc);
    
    if(xmlXPathRegisterNs(xpathCtx,  BAD_CAST "sbml", BAD_CAST "http://www.sbml.org/sbml/level3/version1/core") != 0) {
        throw SedException("Error: unable to register NS with prefix");
    }
    
    if(xmlXPathRegisterNs(xpathCtx,  BAD_CAST "qual", BAD_CAST "http://www.sbml.org/sbml/level3/version1/qual/version1") != 0) {
        throw SedException("Error: unable to register NS with prefix");   
    }

    if(xmlXPathRegisterNs(xpathCtx,  BAD_CAST "math", BAD_CAST "http://www.w3.org/1998/Math/MathML") != 0) {
        throw SedException("Error: unable to register NS with prefix");   
    }
}

XMLPatcher::~XMLPatcher()
{        
    
    /* Cleanup xpath object */
    xmlXPathFreeContext(xpathCtx); 
    
    /* Free the document */
    xmlFreeDoc(doc); 

    /* Shutdown libxml */
    xmlCleanupParser();
    
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();
}

void XMLPatcher::changeXML(std::string xpath, std::string new_xml)
{
    xmlDocPtr newDoc = xmlReadMemory(
        new_xml.c_str(), new_xml.size(),
        NULL, NULL, 0
    );
    xmlNodePtr newNode = xmlDocCopyNode(
        xmlDocGetRootElement(newDoc),
        doc,
        1);
      xmlFreeDoc(newDoc);
      if (!newNode) throw SedException("Error: unable to copy node");

    
    
    const xmlChar *xpathExpr = (const xmlChar *) xpath.c_str();
    /* Evaluate xpath expression */
    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(xpathExpr, xpathCtx);
    if(xpathObj == NULL) {
        fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", xpathExpr);
        xmlXPathFreeContext(xpathCtx); 
        xmlFreeDoc(doc); 
        throw SedException("Error: unable to evaluate xpath expression: " + xpath);
    } 
    
    xmlNodeSetPtr nodeset = xpathObj->nodesetval;
    int size = (nodeset) ? nodeset->nodeNr : 0;
    // std::cout << "XPath size = " << size << std::endl;
    /*
     * NOTE: the nodes are processed in reverse order, i.e. reverse document
     *       order because xmlNodeSetContent can actually free up descendant
     *       of the node and such nodes may have been selected too ! Handling
     *       in reverse order ensure that descendant are accessed first, before
     *       they get removed. Mixing XPath and modifications on a tree must be
     *       done carefully !
     */
    for(int i = size - 1; i >= 0; i--) {
        xmlNodePtr node = nodeset->nodeTab[i];
        
        // std::cout << "Node type = " << node->type << std::endl;
        // std::cout << "Node name = " << node->name << std::endl;
        
        xmlNodePtr parent = node->parent;
        
        xmlUnlinkNode(node);
        xmlFreeNode(node);
      xmlNodePtr addedNode = xmlAddChildList(parent, newNode);
      if (!addedNode) {
        // xmlFreeNode(newNode);
        throw SedException("Error: unable to add node");
      }
    
    
    }
}

void XMLPatcher::removeXML(std::string xpath)
{
    const xmlChar *xpathExpr = (const xmlChar *) xpath.c_str();
    /* Evaluate xpath expression */
    
    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(xpathExpr, xpathCtx);
    if(xpathObj == NULL) {
        fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", xpathExpr);
        xmlXPathFreeContext(xpathCtx); 
        xmlFreeDoc(doc); 
        throw SedException("Error: unable to evaluate xpath expression: " + xpath);
    } 
    
    xmlNodeSetPtr nodeset = xpathObj->nodesetval;
    
    int size = (nodeset) ? nodeset->nodeNr : 0;
    for(int i = size - 1; i >= 0; i--) 
    {
        xmlNodePtr node = nodeset->nodeTab[i];    
        xmlUnlinkNode(node);
        xmlFreeNode(node);
    }
}

void XMLPatcher::addXML(std::string xpath, std::string new_xml)
{
    xmlDocPtr newDoc = xmlReadMemory(
        new_xml.c_str(), new_xml.size(),
        NULL, NULL, 0
    );
    xmlNodePtr newNode = xmlDocCopyNode(
        xmlDocGetRootElement(newDoc),
        doc,
        1);
      xmlFreeDoc(newDoc);
      if (!newNode) throw SedException("Error: unable to copy node");

    
    
    const xmlChar *xpathExpr = (const xmlChar *) xpath.c_str();
    /* Evaluate xpath expression */
    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(xpathExpr, xpathCtx);
    if(xpathObj == NULL) {
        fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", xpathExpr);
        xmlXPathFreeContext(xpathCtx); 
        xmlFreeDoc(doc); 
        throw SedException("Error: unable to evaluate xpath expression");
    } 
    
    xmlNodeSetPtr nodeset = xpathObj->nodesetval;
    int size = (nodeset) ? nodeset->nodeNr : 0;
    for(int i = size - 1; i >= 0; i--) 
    {
      xmlNodePtr node = nodeset->nodeTab[i];
      xmlNodePtr addedNode = xmlAddChildList(node, newNode);
      if (!addedNode) {
        xmlFreeNode(newNode);
        throw SedException("Error: unable to add node");
      }
    }
}

std::string XMLPatcher::getXML()
{
    std::string out;
    xmlChar *s;
    int size;
    xmlDocDumpMemory(doc, &s, &size);
    if (s == NULL)
        throw std::bad_alloc();
    try {
        out = (char *)s;
    } catch (...) {
        xmlFree(s);
        throw;
    }
    xmlFree(s);
    
    return out;
}