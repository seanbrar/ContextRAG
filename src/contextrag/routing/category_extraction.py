import re


def extract_categories_from_file(file_path):
    """Extract unique categories from a file containing category listings.

    The function looks for lines containing "| Categories: " followed by comma-separated
    categories. It extracts these categories, removes duplicates, and returns them as
    a single comma-separated string.

    Args:
        file_path (str): Path to the file containing category listings.

    Returns:
        tuple: A tuple containing:
            - str: Comma-separated string of unique categories, or error message if file not found
            - int: Number of unique categories found (0 if file not found)
    """
    # Regular expression to find categories
    pattern = r"\| Categories: (.+?)\n"

    try:
        with open(file_path, "r") as file:
            text = file.read()

        # Extract categories using the regular expression
        extracted = re.findall(pattern, text)

        # Flatten the list of categories and remove any possible duplicates
        categories = ", ".join(extracted).split(", ")
        unique_categories = list(set(categories))

        # Join the unique categories into a single string
        result = ", ".join(unique_categories)
        return result, len(unique_categories)
    except FileNotFoundError:
        return "File not found. Please check the file path.", 0


if __name__ == "__main__":
    file_path = "output_2024-01-11T05-16-42.txt"
    categories, total_number = extract_categories_from_file(file_path)

    print("Categories:", categories)
    print("Total Number of Categories:", total_number)
