# First, let's create a layout for the houses


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
