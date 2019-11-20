import numpy, os, sys

def get_raw_data(file):
    with open(file, 'r') as f:
        raw_lines = f.readlines()
        
        first_index = next(i for i, col in enumerate(raw_lines[0].strip('\n').split('\t')) if col == "State")
        raw_data = [line.strip('\n').split('\t') for line in raw_lines[1:]]
        
        raw_states = [[s for s in t_data[first_index::3]] for t_data in raw_data]
        raw_probas = [[s for s in t_data[first_index+1::3]] for t_data in raw_data]
        raw_errors = [[s for s in t_data[first_index+2::3]] for t_data in raw_data]
        
        indexes = [float(t_data[0]) for t_data in raw_data]
        states = set()
        for t_states in raw_states:
            states.update(t_states)
        states = sorted(list(states))
        states_indexes = {state:index for index, state in enumerate(states)}
        
        new_data = numpy.zeros((len(raw_probas), len(states)))
        
        for i, t_probas in enumerate(raw_probas):
            for j, proba in enumerate(t_probas):
                new_data[i, states_indexes[raw_states[i][j]]] = proba
            
        return new_data
    
def main(args):
    if len(args) >= 3 and os.path.exists(args[1]) and os.path.exists(args[2]) and args[1].endswith(".csv") and args[2].endswith(".csv"):
        table1 = get_raw_data(args[1])
        table2 = get_raw_data(args[2])

        if len(args) > 3 and args[3] == "--exact":
            sys.exit(0 if numpy.all(table1 == table2) else 1)
        else:
            sys.exit(0 if numpy.all(numpy.isclose(table1, table2, rtol=1e-4, atol=1e-8)) else 1)
    else:
        print("Wrong argument number, or files do not exist")    
        sys.exit(1)
    
if __name__ == '__main__':
    main(sys.argv)