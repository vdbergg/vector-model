import math
import re

collection = []


def get_vocabulary():
    vocabulary = []
    for doc in collection:
        [vocabulary.append(term) for term in doc if term not in vocabulary]

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
    return round(math.log(n) - math.log(nt), 2)


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

    quotient = doc_norm * query_norm

    if quotient == 0:
        return 0

    return round(inner_product / quotient, 2)


def provide_rank(query):
    global collection
    collection = []

    parser_collection()

    vectors_weight_docs = get_vectors_weight_docs()
    vector_weight_query = get_vector_weight_query(query.split())

    rank = [cosine(v, vector_weight_query) for v in vectors_weight_docs]

    rank = [(i, ranked) for i, ranked in enumerate(rank) if ranked != 0]

    return sorted(rank, key=lambda tup: tup[1])


def parser_collection():
    doc_paths = [
        '/home/berg/PycharmProjects/vector-model/documents/doc1.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc2.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc3.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc4.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc5.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc6.txt',
        '/home/berg/PycharmProjects/vector-model/documents/doc7.txt',
    ]

    for doc_path in doc_paths:
        file = open(doc_path, 'r')
        terms = file.read()
        collection.append(
            [
                (
                    re.sub(r'[?|%|#|/|$|.|"|:|;|!|,|\n|(|)|[|]|{|}|0-9|]', r'', term)
                ).lower() for term in terms.split(' ')
            ]
        )

    for i, doc in enumerate(collection):
        collection[i] = [re.sub('^[0-9]+', '', term.replace('-', '')) for term in doc]
        collection[i] = [term for term in collection[i] if term is not '']

    return collection


def run():
    query = input('Enter a query: ')

    ranked = provide_rank(query)

    if not ranked:
        print('Nenhum resultado encontrado')
    else:
        print(ranked)

    run()


run()

