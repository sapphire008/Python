"""Graph tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.processing.graph import (connected_components, 
    _to_tuples, _to_list)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _clip(x, m, M):
    return [_ for _ in x if m <= _ < M]

n = 5
probe_adjacency_list = {i: set(_clip([i-1, i+1], 0, n))
            for i in range(n)}

CHUNK = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ]
            
def _assert_components(chunk, components, **kwargs):
    if not isinstance(chunk, np.ndarray):
        chunk = np.array(chunk)
    chunk_strong = kwargs.pop('chunk_strong', None)
    if chunk_strong is not None and not isinstance(chunk_strong, np.ndarray):
        chunk_strong = np.array(chunk_strong)
    comp = connected_components(chunk, probe_adjacency_list=probe_adjacency_list, 
                                chunk_strong=chunk_strong, return_objects=False,
                                **kwargs)   
    assert len(comp) == len(components), (len(comp), len(components))
    for c1, c2 in zip(comp, components):
        assert set(c1) == set(c2)
    
    
# -----------------------------------------------------------------------------
# Tests A: 1 time step, 1 element
# -----------------------------------------------------------------------------
def test_components_A_1():
    _assert_components([
            [0, 0, 0, 0, 0],
        ],  [])
        
def test_components_A_2():
    _assert_components([
            [1, 0, 0, 0, 0],
        ],  [[(0, 0)]])
        
def test_components_A_3():
    _assert_components([
            [0, 1, 0, 0, 0],
        ],  [[(0, 1)]])
        
def test_components_A_4():
    _assert_components([
            [0, 0, 0, 1, 0],
        ],  [[(0, 3)]])
        
def test_components_A_5():
    _assert_components([
            [0, 0, 0, 0, 1],
        ],  [[(0, 4)]])
        
    
# -----------------------------------------------------------------------------
# Tests B: 1 time step, 2 elements
# -----------------------------------------------------------------------------
def test_components_B_1():
    _assert_components([
            [1, 1, 0, 0, 0],
        ],  [[(0, 0), (0, 1)]])
        
def test_components_B_2():
    _assert_components([
            [1, 0, 1, 0, 0],
        ],  [[(0, 0)], [(0, 2)]])
        
def test_components_B_3():
    _assert_components([
            [1, 0, 0, 0, 1],
        ],  [[(0, 0)], [(0, 4)]])
        
def test_components_B_4():
    _assert_components([
            [0, 1, 0, 1, 0],
        ],  [[(0, 1)], [(0, 3)]])
        
    
# -----------------------------------------------------------------------------
# Tests C: 1 time step, 3 elements
# -----------------------------------------------------------------------------
def test_components_C_1():
    _assert_components([
            [1, 1, 1, 0, 0],
        ],  [[(0, 0), (0, 1), (0, 2)]])
        
def test_components_C_2():
    _assert_components([
            [1, 1, 0, 1, 0],
        ],  [[(0, 0), (0, 1)], [(0, 3)]])
        
def test_components_C_3():
    _assert_components([
            [1, 0, 1, 1, 0],
        ],  [[(0, 0)], [(0, 2), (0, 3)]])
        
def test_components_C_4():
    _assert_components([
            [0, 1, 1, 1, 0],
        ],  [[(0, 1), (0, 2), (0, 3)]])
        
def test_components_C_5():
    _assert_components([
            [0, 1, 1, 0, 1],
        ],  [[(0, 1), (0, 2)], [(0, 4)]])
        
    
# -----------------------------------------------------------------------------
# Tests D: 5 time steps, varying connected_component_join_size
# -----------------------------------------------------------------------------
def test_components_D_1():
    _assert_components([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ],  [[(1, 2)], 
             [(2, 0)], [(2, 2), (2, 3)],
             [(3, 0)], [(3, 3)],
             [(4, 1)], [(4, 3), (4, 4)],
             ])
        
def test_components_D_2():
    _assert_components([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ],  [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)], 
             [(2, 0), (3, 0), (4, 1)],
             ],
        connected_component_join_size=1
        )
        
def test_components_D_3():
    _assert_components(CHUNK,  
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), 
              (2, 0), (3, 0), (4, 1)],
             ],
        connected_component_join_size=2
        )
        

# -----------------------------------------------------------------------------
# Tests E: 5 time steps, strong != weak
# -----------------------------------------------------------------------------
def test_components_E_1():
    _assert_components(CHUNK,
            [],
        connected_component_join_size=0,
        chunk_strong=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )
        
def test_components_E_2():
    _assert_components(CHUNK,
            [[(1, 2)], 
             ],
        connected_component_join_size=0,
        chunk_strong=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )
        
def test_components_E_3():
    _assert_components(CHUNK,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)],
             ],
        connected_component_join_size=1,
        chunk_strong=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ]
        )
        
def test_components_E_4():
    _assert_components(CHUNK,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), 
              (2, 0), (3, 0), (4, 1)],
             ],
        connected_component_join_size=2,
        chunk_strong=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )
        
def test_components_E_5():
    _assert_components(CHUNK,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), 
              (2, 0), (3, 0), (4, 1)],
             ],
        connected_component_join_size=2,
        chunk_strong=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
        )
        