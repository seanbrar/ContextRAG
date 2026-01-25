"""Example script demonstrating chromaroute VectorStore usage."""

from chromaroute import VectorStore


def main() -> None:
    vector_store = VectorStore(collection_name="my_collection")

    documents = [
        "The capital of California is Sacramento.",
        "Sacramento is known for its vibrant farm-to-fork dining scene, with numerous restaurants sourcing ingredients from the surrounding agricultural region.",
        "The capital of Texas is Austin.",
        'Austin is famous for its live music scene, earning it the nickname "Live Music Capital of the World."',
        "The capital of Florida is Tallahassee.",
        "Tallahassee is home to the National High Magnetic Field Laboratory, which houses the world's most powerful magnets.",
        "The capital of New York is Albany.",
        "Albany is known for its rich history and architecture, including the Empire State Plaza and the New York State Capitol.",
        "The capital of Illinois is Springfield.",
        "Springfield is famous for its association with Abraham Lincoln, who lived there before becoming the 16th President of the United States.",
        "The capital of Georgia is Atlanta.",
        "Atlanta played a crucial role in the civil rights movement and is home to the Martin Luther King Jr. National Historical Park.",
        "The capital of Ohio is Columbus.",
        "Columbus is known for its vibrant arts scene, including the Columbus Museum of Art and the Wexner Center for the Arts.",
        "The capital of Virginia is Richmond.",
        "Richmond is famous for its Civil War history, with numerous battlefields and historical sites in the area.",
        "The capital of Colorado is Denver.",
        "Denver is known for its proximity to the Rocky Mountains, offering outdoor recreation opportunities such as skiing and hiking.",
        "The capital of Washington is Olympia.",
        "Olympia is home to the Hands On Children's Museum, which features interactive exhibits for children to learn through play.",
    ]

    vector_store.add_documents(documents)

    query_text = "What is the capital of California?"
    results = vector_store.query(query_texts=[query_text])

    print(f"Query: {query_text}")
    for i, result in enumerate(results["documents"][0]):
        print(f"Result {i+1}: {result}")
        print(f"Distance: {results['distances'][0][i]}\n")


if __name__ == "__main__":
    main()
