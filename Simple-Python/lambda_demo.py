def transform_list(nums_list,transform_item):
    transformed_0=transform_item(nums_list[0])
    transformed_1=transform_item(nums_list[1])
    return [transformed_0, transformed_1]

print(transform_list([2,3],lambda num: num**2))
