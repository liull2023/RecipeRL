episode_length = 32
top_k = (1, )
device = 'cuda:0'

State_compare_flag = [False,False,False, False, False, 
                        False,False,False,False, True, 
                    ]

linear_input_dim_mapping = {
    (0, 1, 2): 1000,
    (3, 4): 2000,
    (5, 6, 7): 1100,
    (8, 9): 2100
}

linear_input_dim = None

for flags, dim in linear_input_dim_mapping.items():
    if any(State_compare_flag[i] for i in flags):
        linear_input_dim = dim
        break