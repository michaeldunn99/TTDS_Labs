import re
import sys
from collections import defaultdict
import Stemmer
import numpy as np


#Function to read in stop words to a set
def read_in_stop_words_to_set(file_name):
    set_of_words = set()
    with open(file_name,'r') as file:
        for line in file:
            line = re.sub(r"\n", "", line)
            set_of_words.add(line)
    return set_of_words

#Define set of stop words
default_stop_words = read_in_stop_words_to_set('data/ttds_2023_english_stop_words.txt')

#Instantiate an instance from a porterstemmer class
default_stemmer = Stemmer.Stemmer('english')

#Define my default porterstemmer function
default_porter_stemmer_function = default_stemmer.stemWord


def convert_non_alphanumeric_to_space(text_input):
    """
    Replace all non-alphanumeric and non-space characters from the given text input with a space.

    Args:
        text_input (str): The input string from which non-alphanumeric characters are to be replaced.

    Returns:
        str: The resulting string after replacing all non-alphanumeric characters with space characters.
    """
    token_pattern = r"[^a-z\s]"
    return re.sub(token_pattern, " ", text_input)


def remove_stop_words_function(iterable_of_text_words, iterable_of_stop_words=default_stop_words):
    """
    Remove stop words from a given iterable of text words.

    Args:
        iterable_of_text_words (iterable): An iterable containing words from which stop words need to be removed.
        iterable_of_stop_words (iterable): An iterable containing stop words that need to be removed from the text words.

    Returns:
        str: A list of text_words with the stop words removed from the original list.
    """
    return [word for word in iterable_of_text_words if word not in iterable_of_stop_words]

def my_porter_stemmer(iterable_of_text_words, my_porter_stemmer_function=default_porter_stemmer_function):
    """
    Applies a given Porter stemmer function to each word in an iterable of text words.

    Args:
        iterable_of_text_words (iterable): An iterable containing words to be stemmed.
        my_porter_stemmer_function (function): A function that takes a word as input and returns its stemmed form.

    Returns:
        list: A list of stemmed words.
    """
    return [my_porter_stemmer_function(word) for word in iterable_of_text_words]




class Corpus:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus_dict = defaultdict(lambda: {"docs": set(), "N11": 0, "N10": 0, "N01": 0, "N00": 0, "MI":0, "Chi_Sq":0})
        self.corpus_doc_length = 0
        self.load_corpus()

    def load_corpus(self):
        doc_id = 0
        try:
            with open(self.file_path, 'r') as corpus_file:
                for line in corpus_file:
                    if line == "\n":
                        doc_id += 1
                        # self.corpus_dict["_total_docs"] += 1
                        continue
                    processed_line = self.my_preprocessor(line)
                    for word in processed_line:
                        if word == "her":
                            print("banter")
                        self.corpus_dict[word]["docs"].add(doc_id)
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        #Update to only include words in 10 or more documents
        self.corpus_dict = {word: data for word, data in self.corpus_dict.items() if len(data["docs"]) >= 10}
        unique_docs = {doc_id for data in self.corpus_dict.values() for doc_id in data["docs"]}
        self.corpus_doc_length = len(unique_docs)
        

    def my_preprocessor(self, text_line):
        """
        Preprocess the text line by converting to lowercase, removing stop words, and applying Porter stemming.
        """
        processed_text = text_line.lower()
        processed_text = processed_text.split()
        processed_text = remove_stop_words_function(processed_text)
        processed_text = " ".join(processed_text)
        processed_text = convert_non_alphanumeric_to_space(processed_text)
        processed_text = processed_text.split()
        processed_text = my_porter_stemmer(processed_text)
        processed_text = remove_stop_words_function(processed_text)
        return processed_text

    def update_counts(self, other_corpus):
        corpus1_total = self.corpus_doc_length
        corpus2_total = other_corpus.corpus_doc_length

        for word in self.corpus_dict:
            if word == "_total_docs":
                continue
            current_dict = self.corpus_dict[word]
            current_dict["N11"] = len(current_dict["docs"])
            current_dict["N01"] = corpus1_total - current_dict["N11"]
            if word in other_corpus.corpus_dict:
                current_dict["N10"] = len(other_corpus.corpus_dict[word]["docs"])
            else:
                current_dict["N10"] = 0  # or any default value you prefer
            current_dict["N00"] = corpus2_total - current_dict["N10"]

    def mutual_inf(self, word, other_corpus):
        word_dict = self.corpus_dict[word]
        corpus1_total = self.corpus_doc_length
        corpus2_total = other_corpus.corpus_doc_length
        overall_total = corpus1_total + corpus2_total

        N11 = word_dict["N11"]
        N10 = word_dict["N10"]
        N01 = word_dict["N01"]
        N00 = word_dict["N00"]
        N1_dot = N11 + N10
        N_dot_1 = N11 + N01
        N_dot_0 = N10 + N00
        N0_dot = N01 + N00
        
        try: 
            # Calculate mutual information components
            if N11 == 0:
                mi_component_1 = 0
            else:
                mi_component_1 = (N11 / overall_total) * np.log2((overall_total * N11) / (N1_dot * N_dot_1))
            if N10 == 0:
                mi_component_2 = 0
            else:
                mi_component_2 = (N10 / overall_total) * np.log2((overall_total * N10) / (N1_dot * N_dot_0))
            if N01 == 0:
                mi_component_3 = 0
            else:
                mi_component_3 = (N01 / overall_total) * np.log2((overall_total * N01) / (N0_dot * N_dot_1))
            if N00 == 0:
                mi_component_4 = 0
            else:
                mi_component_4 = (N00 / overall_total) * np.log2((overall_total * N00) / (N0_dot * N_dot_0))

            # Sum the components to get the current mutual information
            curr_mutual_inf = mi_component_1 + mi_component_2 + mi_component_3 + mi_component_4
        except ZeroDivisionError:
            sys.exit("Error: Division by zero")
        
        word_dict["MI"] = curr_mutual_inf
    
    def chi_sq(self, word):
        word_dict = self.corpus_dict[word]
        N_11 = word_dict["N11"]
        N_10 = word_dict["N10"]
        N_01 = word_dict["N01"]
        N_00 = word_dict["N00"]

        current_chi_squared = \
            (N_11 + N_10 + N_01 + N_00) * ((N_11 * N_00 - N_10 * N_01)**2) / ((N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00))
        word_dict["Chi_Sq"] = current_chi_squared

def main():
    corpus1_path = 'data/corpus1.txt'
    corpus2_path = 'data/corpus2.txt'

    corpus1 = Corpus(corpus1_path)
    corpus2 = Corpus(corpus2_path)

    corpus1.update_counts(corpus2)
    corpus2.update_counts(corpus1)

    for word in corpus1.corpus_dict:
        if word != "_total_docs":
            corpus1.mutual_inf(word, corpus2)
            corpus1.chi_sq(word)
    
    for word in corpus2.corpus_dict:
        if word != "_total_docs":
            corpus2.mutual_inf(word, corpus1)
            corpus2.chi_sq(word)

    sorted_corpus1_mi = sorted(corpus1.corpus_dict.items(), key=lambda item: item[1]["MI"], reverse=True)
    sorted_corpus1_mi = [(item[0], item[1]["MI"]) for item in sorted_corpus1_mi]
    for item in sorted_corpus1_mi[0:10]:
        print(item)

    print() 
    print()

    sorted_corpus2_mi = sorted(corpus2.corpus_dict.items(), key=lambda item: item[1]["MI"], reverse=True)
    sorted_corpus2_mi = [(item[0], item[1]["MI"]) for item in sorted_corpus2_mi]
    for item in sorted_corpus2_mi[0:10]:
        print(item)

    print() 
    print()

    sorted_corpus1_chi_sq = sorted(corpus1.corpus_dict.items(), key=lambda item: item[1]["Chi_Sq"], reverse=True)
    sorted_corpus1_chi_sq = [(item[0], item[1]["Chi_Sq"]) for item in sorted_corpus1_chi_sq]
    for i in range(10):
        print(sorted_corpus1_chi_sq[i])

    print()
    print()

    sorted_corpus2_chi_sq = sorted(corpus2.corpus_dict.items(), key=lambda item: item[1]["Chi_Sq"], reverse=True)
    sorted_corpus2_chi_sq = [(item[0], item[1]["Chi_Sq"]) for item in sorted_corpus2_chi_sq]
    for item in sorted_corpus2_chi_sq[0:10]:
        print(item)
    

if __name__ == "__main__":
    main()
