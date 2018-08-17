import math

collection = [['A', 'A', 'A', 'B'], ['A', 'A', 'C'], ['A', 'A'], ['B', 'B']]


def get_vocabulary():
    vocabulary = []
    for doc in collection:
        for term in doc:
            [vocabulary.append(letter) for letter in term if letter not in vocabulary]

    return sorted(vocabulary)


def get_total_docs():
    return len(collection)


def get_occurrence_term_in_docs(term):
    occurrences = [doc for doc in collection if term in doc]
    return len(occurrences)


def get_tf(doc, t):
    occurrences = [term for term in doc if term == t]
    return len(occurrences)


def get_idf(term):
    n = get_total_docs()
    nt = get_occurrence_term_in_docs(term)
    return round(math.log(n/nt), 2)


def get_weight(doc, term):
    return round(get_idf(term) * get_tf(doc, term), 2)


def get_vectors_weight_docs():
    vocabulary = get_vocabulary()
    vectors = []

    for doc in collection:
        vectors.append([get_weight(doc, term) for term in vocabulary])

    return vectors


def get_vector_weight_query(query):
    vocabulary = get_vocabulary()
    return [get_weight(query, term) for term in vocabulary]


def get_inner_product(u, v):
    return sum([a * b for (a, b) in zip(u, v)])


def get_norm(v):
    return round(math.sqrt(sum([i * i for i in v])), 2)


def cosine(doc, query):
    inner_product = get_inner_product(doc, query)

    doc_norm = get_norm(doc)
    query_norm = get_norm(query)

    return round(inner_product / (doc_norm * query_norm), 2)


def provide_rank(query):
    vectors_weight_docs = get_vectors_weight_docs()
    vector_weight_query = get_vector_weight_query(query.split())

    return [cosine(v, vector_weight_query) for v in vectors_weight_docs]


print(provide_rank('A A C'))
