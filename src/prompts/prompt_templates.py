def house_info_layout(houses):
    # Create an empty string
    layout = ""
    # Iterate over the houses
    for house in houses:
        # For each house, append the information to the string using f-strings
        # The following way using brackets is a good way to make the code readable as in each line you can start a new f-string that will appended to the previous one
        layout += (
            f"House located at {house['address']}, {house['city']}, {house['state']} {house['zip']} with "
            f"{house['bedrooms']} bedrooms, {house['bathrooms']} bathrooms, "
            f"{house['square_feet']} sq ft area, priced at ${house['price']}, "
            f"built in {house['year_built']}.\n"
        )  # Don't forget the new line character at the end!
    return layout


def generate_prompt(query, houses):
    # The code made above is modular enough to accept any list of houses, so you could also choose a subset of the dataset.
    # This might be useful in a more complex context where you want to give only some information to the LLM and not the entire data
    houses_layout = house_info_layout(houses)
    # Create a hard-coded prompt. You can use three double quotes (") in this cases, so you don't need to worry too much about using single or double quotes and breaking the code
    PROMPT = f"Use the following houses information to answer users queries.\n \
                {houses_layout} \
                Query: {query}"
    return PROMPT
