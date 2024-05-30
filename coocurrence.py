import pickle
from multiprocessing import Process, Queue
from os import getpid

from scipy.sparse import lil_matrix
from tqdm import tqdm

from utils import concat_iterables, get_data_iterator_from
from tokenizer import load_tokenizer


def calculate_word_word_coocurrence(
    filtered_doc: list[int], voc_size: int, window_size: int
) -> lil_matrix:
    word_word_coocurrence = lil_matrix((voc_size, voc_size))

    for center_index in range(len(filtered_doc)):
        left_window_index = max(0, center_index - window_size)
        right_window_index = min(len(filtered_doc) - 1, center_index + window_size)
        context_indexes = concat_iterables(
            range(left_window_index, center_index),
            range(center_index + 1, right_window_index + 1),
        )

        for context_index in context_indexes:
            center_tok = filtered_doc[center_index]
            context_tok = filtered_doc[context_index]
            word_word_coocurrence[center_tok, context_tok] += 1

    return word_word_coocurrence


def singleprocess_cooc(huggingface_url: str, window_size: int = 2, max_docs: int = -1) -> lil_matrix:
    tokenizer = load_tokenizer()
    unk_token = tokenizer.encode("[UNK]").ids
    vocab_size = tokenizer.get_vocab_size()

    cooc_matrix = lil_matrix((vocab_size, vocab_size))

    def filter_unk_token(unfiltered_doc):
        return [tok for tok in unfiltered_doc if tok != unk_token]

    for doc_count, doc in tqdm(enumerate(get_data_iterator_from(huggingface_url)), desc="doc"):
        if doc_count == max_docs:
            break
        encoded_doc = tokenizer.encode(doc).ids
        filtered_encoded_doc = filter_unk_token(encoded_doc)
        word_word_cooccurrence_mat = calculate_word_word_coocurrence(
            filtered_doc=filtered_encoded_doc,
            voc_size=vocab_size,
            window_size=window_size,
        )
        cooc_matrix += word_word_cooccurrence_mat

    with open("./data/cooc_matrix_lil.pk", "wb") as file:
        pickle.dump(file, cooc_matrix)

    return cooc_matrix


def worker(
    input_queue: Queue, output_queue: Queue, vocab_size: int, window_size: int
) -> int:
    """
    Gets a list of tokens and computes the word-word coocurrences based on a tokenized string.
    Does the heavy lifting by:

    1. Step: input_queue -> [1, 5, 2, 3, 0, 4]
    2. Step: word-word cooccurrences with window_size=2 in the form of "(center_token, context_token)" -> (1,5), (1,2), (5,1), (5,2), (5,3), (2,1), (2,3), (2,0), (3,5), (3,2), (3,0), (3,4), (0,2), (0,3), (0,4), (4,3), (4,0)
    3. Step: word-word cooccurrence matrix -> [
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0]
    ]
    Note: The word-word coccurrence matrix not only holds 0s and 1s, but also higher count since word that commonly occurr in their neighbourhood are presented that way.
    Although some occurrences happen more frequently, most of pairings will not large vocabular size.
    This motivates the usage of a scipy.sparse matrix.
    Due to its advantages performance for frquently updates entries the lil_matrix format is used.
    """

    word_word_coocurrence = lil_matrix((vocab_size, vocab_size))
    doc_count = 0

    print(f"Worker ({getpid()}): Starting", flush=True)
    while True:
        # encoded_doc can be either "END" (string) or [2, 3, 0, ...] (list of token)
        encoded_doc = input_queue.get()

        # receive the ending signal, there are no more entries in the queue to process
        if isinstance(encoded_doc, str) and encoded_doc == "END":
            print(f"Worker ({getpid()}): sending partial result", flush=True)
            output_queue.put(word_word_coocurrence)
            print(f"Worker ({getpid()}): sending end signal", flush=True)
            output_queue.put("END")
            break

        doc_count += 1

        # make sure the worker is not stuck
        if doc_count % 1000 == 0:
            print(f"Worker ({getpid()}): processed {doc_count} docs", flush=True)

        # finding the word-word cooccurrences for the whole document
        for center_index in range(len(encoded_doc)):
            left_window_index = max(0, center_index - window_size)
            right_window_index = min(len(encoded_doc) - 1, center_index + window_size)
            context_indexes = concat_iterables(
                range(left_window_index, center_index),
                range(center_index + 1, right_window_index + 1),
            )
            for context_index in context_indexes:
                center_tok = encoded_doc[center_index]
                context_tok = encoded_doc[context_index]
                word_word_coocurrence[center_tok, context_tok] += 1

    # make sure the worker ended
    print(f"Worker ({getpid()}): Ending", flush=True)
    return 0  # process end with exitcode 0


def feeder(queue: Queue, huggingface_url: str, max_docs: int, worker_count: int) -> int:
    """
    Loading and pre-processing the documents from disk.
    For memory efficiency generator objects are used.

    1. Step: load document from disk (already filtered out stopwords, digits and special characters)
    2. Step: tokenize document string
    3. Step: filter unkown token
    4. Step: put filtered tokenized document in the queue
    """
    tokenizer = load_tokenizer()
    unk_token = tokenizer.encode("[UNK]").ids

    def filter_unk_token(unfiltered_doc: list[int]) -> list[int]:
        return [tok for tok in unfiltered_doc if tok != unk_token]

    print(f"Feeder ({getpid()}): Starting", flush=True)
    for doc_count, doc in enumerate(get_data_iterator_from(huggingface_url)):
        # limiting the amount of docs processed
        if doc_count == max_docs:
            break
        if doc_count % 1000 == 0:
            print(f"Feeder ({getpid()}): processed {doc_count} docs", flush=True)
        encoded_doc = tokenizer.encode(doc).ids
        filtered_encoded_doc = filter_unk_token(encoded_doc)

        queue.put(filtered_encoded_doc)

    # workers can quit when queue is worked through
    print(f"Feeder ({getpid()}): Adding END marker to worker queue", flush=True)
    for _ in range(worker_count):
        queue.put("END")

    print(f"Feeder ({getpid()}): Ending", flush=True)
    return 0


def collector(vocab_size: int, worker_count: int, collector_queue: Queue) -> lil_matrix:
    """
    Collecting the partial cooccurrence matrices from the workers and combining them into a singular matrix.

    1. Step: create resulting matrix (word_word_cooccurrence)
    2. Step: get partial matrix from worker (collector_queue.get())
    3. Step: add partial matrix to resulting matrix
    4. Step: goto Step 2.
    """
    print(f"Collector ({getpid()}): Start collecting", flush=True)
    word_word_cooccurrence = lil_matrix((vocab_size, vocab_size))
    collected = 0
    while True:
        partial_word_word_cooccurrence = collector_queue.get()
        # stop collecting when all workers send their partial result
        if (
            isinstance(partial_word_word_cooccurrence, str)
            and partial_word_word_cooccurrence == "END"
        ):
            if collected == worker_count:
                break
            else:
                continue

        print(f"Collector ({getpid()}): Collecting part {collected}", flush=True)
        word_word_cooccurrence += partial_word_word_cooccurrence
        collected += 1
        print(f"Collector ({getpid()}): Collected {collected} part(s)", flush=True)

    return word_word_cooccurrence


def multiprocess_cooc(
    huggingface_url: str, window_size: int = 2, max_docs: int = 10000, worker_count: int = 1
):
    tok = load_tokenizer()
    vocab_size = tok.get_vocab_size()
    processes = []

    worker_queue = Queue(5000)
    collector_queue = Queue()

    # init feeder
    print(f"Main ({getpid()}): Creating feeder", flush=True)
    p = Process(target=feeder, args=(worker_queue, huggingface_url, max_docs, worker_count))
    processes.append(p)

    # init workers
    print(f"Main ({getpid()}): Creating workers", flush=True)
    for _ in range(worker_count):
        p = Process(
            target=worker, args=(worker_queue, collector_queue, vocab_size, window_size)
        )
        processes.append(p)

    # starting all processes
    for process in processes:
        process.start()

    word_word_cooccurrence = collector(
        vocab_size=vocab_size, worker_count=worker_count, collector_queue=collector_queue
    )
    with open("./data/multiprocess_cooc_matrix_lil.pk", "wb") as file:
        pickle.dump(word_word_cooccurrence, file)

    # when joining processes before collecting, deadlock is immanent because workers return lil_matrixes are too large
    for p in processes:
        p.join()
        print(f"Main ({getpid()}): joined Process {p.pid}", flush=True)
