def dict_to_args(my_dict):
    args_list = []
    for k, v in my_dict.items():
        k = '--' + k if len(k) > 1 else '-' + k
        args_list.append(k)
        if v is not True:  # no need to specify a value for store_true/store_const options
            args_list.append(str(v))
    return args_list
