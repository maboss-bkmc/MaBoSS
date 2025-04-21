#ifndef __OBSERVED_GRAPH_H__
#define __OBSERVED_GRAPH_H__

#include <ostream>
#include "Network.h"

#ifdef PYTHON_API
#include <Python.h>
#endif

class ObservedGraph {
    std::map<NetworkState_Impl, std::map<NetworkState_Impl, unsigned int>> counts;
    std::map<NetworkState_Impl, std::map<NetworkState_Impl, double>> durations;
    NetworkState_Impl graph_mask;
    std::vector<const Node*> graph_nodes;
    std::vector<NetworkState_Impl> graph_states;
    NetworkState_Impl last_state;
    double last_time;
    
    NetworkState_Impl getObservedState(NetworkState state) const;

public:

    ObservedGraph(const Network* network);  
    void init(unsigned int count=0, double duration=0.0);    
    void addFirstTransition(NetworkState origin_state);    
    void addTransition(NetworkState destination_state, double time);
    const std::map<NetworkState_Impl, std::map<NetworkState_Impl, unsigned int>>& getCounts() const;  
    const std::map<NetworkState_Impl, std::map<NetworkState_Impl, double>>& getDurations() const;  
    size_t size() const;
    std::vector<NetworkState_Impl> getStates() const;
    void display(std::ostream * output_observed_graph, std::ostream * output_observed_durations, const Network* network) const;
    void mergePairOfObservedGraph(const ObservedGraph* observed_graph_2);
    void epilogue();
    
#ifdef MPI_COMPAT
    ObservedGraph(char * buff, unsigned int buff_size);
    
    static void mergePairOfMPIObservedGraph(ObservedGraph* graph, int world_rank, int dest, int origin, bool pack=true);
    unsigned int MPI_Pack_Size_ObservedGraph() const;  
    void MPI_Unpack_ObservedGraph(char* buff, unsigned int buff_size);
    char* MPI_Pack_ObservedGraph(int dest, unsigned int * buff_size) const;
    void MPI_Send_ObservedGraph(int dest) const;
    void MPI_Recv_ObservedGraph(int origin);

#endif

#ifdef PYTHON_API
    PyObject* getNumpyObservedGraph(const Network* network) const;
    PyObject* getNumpyObservedDurations(const Network* network) const;
#endif
};

#endif