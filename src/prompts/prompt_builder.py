from abc import ABC, abstractmethod
from textwrap import dedent


class PromptBuilder(ABC):
    INSTRUCTIONS = """

    PROMPT_TEMPLATE = """

    @abstractmethod
    def build_prompt(self, question, search_results):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def build_context(self, search_results):
        raise NotImplementedError("Subclasses must implement this method.")


class FAQPromptBuilder(PromptBuilder):
    INSTRUCTIONS = dedent("""\
        Your task is to answer questions from the course participants based on the provided context.

        Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, respond with "I don't know."
    """)

    PROMPT_TEMPLATE = dedent("""\
        Question: 
        {question}

        Context: 
        {context}
    """)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.PROMPT_TEMPLATE.format(question=question, context=context)
        return prompt.strip()


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
