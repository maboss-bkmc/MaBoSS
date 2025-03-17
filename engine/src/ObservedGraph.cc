#include "ObservedGraph.h"

#ifdef PYTHON_API
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MABOSS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

NetworkState_Impl ObservedGraph::getObservedState(NetworkState state) const
{
    return state.getState() & graph_mask;
}

ObservedGraph::ObservedGraph(const Network* network) 
{
    NetworkState state_mask;
    for (auto * node : network->getNodes()) {
        if (node->inGraph()) {
            graph_nodes.push_back(node);
            state_mask.flipState(node);
        }
    }
    graph_mask = state_mask.getState();
    
    graph_states.resize((int) pow(2, graph_nodes.size()));

    unsigned int i=0;
    for (auto graph_state : graph_states) {
        NetworkState state(graph_state);
        
        unsigned int j=0;
        for (auto* node: graph_nodes){
        if ((i & (1ULL << j)) > 0)
        {
            state.flipState(node);
        }
        j++;
        }
        
        graph_states[i] = state.getState();
        i++;
    }
}

void ObservedGraph::init(int count, double duration)
{
    for (auto origin_state : graph_states){
        for (auto destination_state: graph_states){
            counts[origin_state][destination_state] = count;
            durations[origin_state][destination_state] = duration;
        }
    }
}
    
void ObservedGraph::addFirstTransition(NetworkState origin_state)
{
    last_state = origin_state.getState() & graph_mask;
    last_time = 0.0;
}

void ObservedGraph::addTransition(NetworkState destination_state, double time)
{
    NetworkState_Impl observed_destination_state = destination_state.getState() & graph_mask;
    if (observed_destination_state != last_state) {
        counts[last_state][observed_destination_state] += 1;
        durations[last_state][observed_destination_state] += time - last_time;
        last_state = observed_destination_state;
        last_time = time;
    }
    
}

const std::map<NetworkState_Impl, std::map<NetworkState_Impl, unsigned int>>& ObservedGraph::getCounts() const
{
    return counts;
}

const std::map<NetworkState_Impl, std::map<NetworkState_Impl, double>>& ObservedGraph::getDurations() const
{
    return durations;
}

size_t ObservedGraph::size() const
{
    return counts.size();
}

std::vector<NetworkState_Impl> ObservedGraph::getStates() const
{
    return graph_states;
}

void ObservedGraph::display(std::ostream * output_observed_graph, std::ostream * output_observed_durations, const Network* network) const
{
    if (counts.size() > 0)
    {
        (*output_observed_graph) << "State";
        for (auto state: graph_states) {
            (*output_observed_graph) << "\t" << NetworkState(state).getName(network);
        }
        (*output_observed_graph) << std::endl;
        
        for (auto row: counts) {
            (*output_observed_graph) << NetworkState(row.first).getName(network);
        
            for (auto cell: row.second) {
                (*output_observed_graph) << "\t" << cell.second;
            }
            
            (*output_observed_graph) << std::endl;
        }
        
        (*output_observed_durations) << "State";
        for (auto state: graph_states) {
            (*output_observed_durations) << "\t" << NetworkState(state).getName(network);
        }
        (*output_observed_durations) << std::endl;
        
        for (auto row: durations) {
            (*output_observed_durations) << NetworkState(row.first).getName(network);
        
            for (auto cell: row.second) {
                (*output_observed_durations) << "\t" << cell.second;
            }
            
            (*output_observed_durations) << std::endl;
        }
    }
}

void ObservedGraph::mergePairOfObservedGraph(const ObservedGraph* observed_graph_2)
{
    for (auto origin_state: observed_graph_2->getCounts()){
        for (auto destination_state: origin_state.second) {
            counts[origin_state.first][destination_state.first] += destination_state.second;
            durations[origin_state.first][destination_state.first] += observed_graph_2->getDurations().at(origin_state.first).at(destination_state.first);
        }
    }
    
    delete observed_graph_2;
    observed_graph_2 = NULL;
}

void ObservedGraph::epilogue()
{
    for (auto origin_state: counts){
        for (auto destination_state: origin_state.second) {
            if (destination_state.second > 0)
                durations[origin_state.first][destination_state.first] /= destination_state.second;
        }
    }
}

#ifdef MPI_COMPAT
  
ObservedGraph::ObservedGraph(char * buff, unsigned int buff_size) {
    this->MPI_Unpack_ObservedGraph(buff, buff_size);
}

void ObservedGraph::mergePairOfMPIObservedGraph(ObservedGraph* graph, int world_rank, int dest, int origin, bool pack)
{
    if (world_rank == dest) 
    {
        if (pack) {
            unsigned int buff_size = -1;
            MPI_Recv( &buff_size, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
            if (buff_size > 0) {
                char* buff = new char[buff_size];
                MPI_Recv( buff, buff_size, MPI_PACKED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                if (graph == NULL) {
                    graph = new ObservedGraph(buff, buff_size);
                } else {
                    graph->MPI_Unpack_ObservedGraph(buff, buff_size);
                }
                delete [] buff;
            }
            
        } else {
            graph->MPI_Recv_ObservedGraph(origin);
        }
        
    } else if (world_rank == origin) {

        if (pack) {
            unsigned int buff_size = -1;
            char* buff = graph->MPI_Pack_ObservedGraph(dest, &buff_size);      
            MPI_Send(&buff_size, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
            if (buff_size > 0) {    
                MPI_Send( buff, buff_size, MPI_PACKED, dest, 0, MPI_COMM_WORLD); 
                delete [] buff;            
            }
        } else {
            graph->MPI_Send_ObservedGraph(dest);
        }
    }
}
  
unsigned int ObservedGraph::MPI_Pack_Size_ObservedGraph() const
{
    unsigned int pack_size = sizeof(unsigned int);
    for (auto& row: counts) {
        NetworkState s(row.first);
        pack_size += s.my_MPI_Pack_Size();
        pack_size += row.second.size() * (sizeof(unsigned int) + sizeof(double)); 
    }
    return pack_size;
}
  
void ObservedGraph::MPI_Unpack_ObservedGraph(char* buff, unsigned int buff_size)
{
    int position = 0;

    unsigned int size = -1;
    MPI_Unpack(buff, buff_size, &position, &size, 1, MPI_UNSIGNED, MPI_COMM_WORLD);

    if (size > 0) {
        std::vector<NetworkState_Impl> states;
        for (unsigned int i=0; i < size; i++) {
            NetworkState s;
            s.my_MPI_Unpack(buff, buff_size, &position);
            states.push_back(s.getState());
        }
        
        for (auto& row: counts) {
            for (auto& cell: row.second) {
                unsigned int count = -1;
                MPI_Unpack(buff, buff_size, &position, &count, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
                cell.second += count;
                double duration = 0.0;
                MPI_Unpack(buff, buff_size, &position, &duration, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                durations[row.first][cell.first] += duration;
            }
        }
    }
}

char* ObservedGraph::MPI_Pack_ObservedGraph(int dest, unsigned int * buff_size) const
{
    *buff_size = this->MPI_Pack_Size_ObservedGraph();

    char* buff = new char[*buff_size];
    int position = 0;

    unsigned int size = counts.size();
    MPI_Pack(&size, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);


    for (auto& row: counts) {
        NetworkState s(row.first);
        s.my_MPI_Pack(buff, *buff_size, &position);
    }
    for (auto& row: counts) {
        for (auto& cell: row.second) {
            unsigned int count = cell.second;
            MPI_Pack(&count, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);
            double duration = durations.at(row.first).at(cell.first);
            MPI_Pack(&duration, 1, MPI_DOUBLE, buff, *buff_size, &position, MPI_COMM_WORLD);
        }
    }
    
    return buff;
}
  
void ObservedGraph::MPI_Send_ObservedGraph(int dest) const
{
    unsigned int nb_states = counts.size();
    MPI_Send(&nb_states, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);

    if (nb_states > 0) {
        for (auto& row: counts) {
        NetworkState s(row.first);
        s.my_MPI_Send(dest);
        }

        for (auto& row: counts) {
            for (auto& cell: row.second) {
                unsigned int count = cell.second;
                MPI_Send(&count, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
                double duration = durations.at(row.first).at(cell.first);
                MPI_Send(&duration, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
    }
}

void ObservedGraph::MPI_Recv_ObservedGraph(int origin)
{
    unsigned int nb_states = -1;
    MPI_Recv(&nb_states, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (nb_states > 0) {
        std::vector<NetworkState_Impl> states;
        for (unsigned int i=0; i < nb_states; i++) {
        NetworkState s;
        s.my_MPI_Recv(origin);
        states.push_back(s.getState());
        }
        
        for (auto& row: counts) {
            for (auto& cell: row.second) {
                unsigned int count = -1;
                MPI_Recv(&count, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cell.second += count;
                double duration = 0.0;
                MPI_Recv(&duration, 1, MPI_DOUBLE, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                durations[row.first][cell.first] += duration;
            }
        } 
        
    }
}
  
  

#endif


#ifdef PYTHON_API

PyObject* ObservedGraph::getNumpyObservedGraph(const Network* network) const
{
    npy_intp dims[2] = {(npy_intp) this->size(), (npy_intp) this->size()};
    PyArrayObject* graph = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 
    PyObject* states = PyList_New(this->size());

    int i=0;
    for (auto& row: counts) 
    {
        PyList_SetItem(states, i, PyUnicode_FromString(NetworkState(row.first).getName(network).c_str()));
        int j=0;
        for (auto& cell: row.second) {
            void* ptr_val = PyArray_GETPTR2(graph, i, j);

            PyArray_SETITEM(graph, (char*) ptr_val, PyLong_FromUnsignedLong(cell.second));
            j++;
        }
        i++;
    }

    return PyTuple_Pack(2, PyArray_Return(graph), states);
}



PyObject* ObservedGraph::getNumpyObservedDurations(const Network* network) const
{
    npy_intp dims[2] = {(npy_intp) this->size(), (npy_intp) this->size()};
    PyArrayObject* graph = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 
    PyObject* states = PyList_New(this->size());

    int i=0;
    for (auto& row: durations) 
    {
        PyList_SetItem(states, i, PyUnicode_FromString(NetworkState(row.first).getName(network).c_str()));
        int j=0;
        for (auto& cell: row.second) {
            void* ptr_val = PyArray_GETPTR2(graph, i, j);

            PyArray_SETITEM(graph, (char*) ptr_val, PyFloat_FromDouble(cell.second));
            j++;
        }
        i++;
    }

    return PyTuple_Pack(2, PyArray_Return(graph), states);
}
#endif