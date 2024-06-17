

def filter_for_query(name_of_column, filter_list):
    if len(filter_list) == 0:
        return ''
    else:
        # add quotes to string values
        formatted_values = ", ".join([f"'{x}'" if isinstance(x, str) else str(x) for x in filter_list])
        return f"AND {name_of_column} IN ({formatted_values})"


def filter_from_right_item_charachters(filter_list, index_of_first_char, lenght_of_chars, name_of_column='item'):
    if len(filter_list) == 0:
        return ''
    else:
        formatted_values = ", ".join([f"'{x}'" if isinstance(x, str) else str(x) for x in filter_list])
        return f"AND SUBSTRING({name_of_column}, {index_of_first_char + 1}, {lenght_of_chars}) IN ({formatted_values})"