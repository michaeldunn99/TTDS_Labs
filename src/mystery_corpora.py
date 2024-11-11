import re
from collections import defaultdict, OrderedDict
import Stemmer


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

#Main preprocessing functino that combines the preprocessing functions above and defines the order of execution
def my_preprocessor(text_line, remove_stop_words, apply_stemming, stop_words=default_stop_words ,porter_stemmer=default_porter_stemmer_function):
    """
    Preprocesses a given text line by applying several text processing steps.

    Args:
        text_line (str): The input text line to be processed.
        stop_words (set): A set of stop words to be removed from the text.
        porter_stemmer (PorterStemmer): An stem method (function) from an instance of PorterStemmer for stemming words.

    Returns:
        str: The processed text line after converting to lowercase, removing alphanumeric characters,
             removing stop words, and applying Porter stemming.
    """
    
    processed_text = text_line.lower()
    processed_text = processed_text.split()
    processed_text = remove_stop_words_function(processed_text, stop_words)
    processed_text = " ".join(processed_text)
    processed_text = convert_non_alphanumeric_to_space(processed_text)
    processed_text = processed_text.split()
    processed_text = my_porter_stemmer(processed_text, porter_stemmer)

    return processed_text


def main():

    corpus1_dict = defaultdict(lambda: {"docs": {}, "N11":0, "N10":0, "N01":0, "N00":0, "MI": 0, "Chi_Sq":0})
    corpus1_dict["_total_docs"] = 0
    with open('data/corpus1.txt') as corpus1:
        docid=0
        for line in corpus1:
            if line == "\n":
                docid +=1
                corpus1_dict["_total_docs"] +=1
                continue
            else:
                processed_line = my_preprocessor(line)
                for word in processed_line:
                    corpus1_dict[word]["docs"].add(docid)
    
    corpus2_dict = defaultdict(lambda: {"docs": {}, "N11":0, "N10":0, "N01":0, "N00":0})
    corpus2_dict["_total_docs"] = 0
    with open('data/corpus2.txt') as corpus2:
        docid2=0
        for line in corpus2:
            if line == "\n":
                docid2 +=1
                corpus2_dict["_total_docs"] +=1
                continue
            else:
                processed_line = my_preprocessor(line)
                for word in processed_line:
                    corpus2_dict[word]["docs"].add(docid)
    
    corpus1_total = corpus1_dict["_total_docs"]
    corpus2_total = corpus2_dict["_total_docs"]
    for word in corpus1_dict:
        current_dict = corpus1_dict[word]
        current_dict["N11"] = len(current_dict["docs"])
        current_dict["N01"] = corpus1_total - current_dict["N11"]
        current_dict["N10"] = len(corpus2_dict[word]["docs"])
        current_dict["N00"] = corpus2_total - current_dict["N10"]
    
    for word in corpus2_dict:
        current_dict = corpus2_dict[word]
        current_dict["N11"] = len(current_dict["docs"])
        current_dict["N01"] = corpus2_total - current_dict["N11"]
        current_dict["N10"] = len(corpus1_dict[word]["docs"])
        current_dict["N00"] = corpus1_total - current_dict["N10"]
    
    def mutual_inf(word_dict):
        
        overall_total = corpus1_total + corpus2_total

        N_11 = word_dict["N11"], N_10 = word_dict["N10"]
        N_01 = word_dict["N10"], N_00 = word_dict["N00"]
        N_1_dot = N_11 + N_10
        N_dot_1 = N_11 + N_01
        N_dot_0 = N_10 + N_00
        N_0_dot = N_01 + N_00

        
        curr_mutual_inf = \
            (N_11 / overall_total) * ((overall_total * N_11) / (N_1_dot * N_dot_1)) + (N_10 / overall_total) * ((overall_total * N_10) / (N_1_dot * N_dot_0)) + (N_01 / overall_total) * ((overall_total * N_01) / (N_0_dot * N_dot_1)) + (N_00 / overall_total) * ((overall_total * N_00) / (N_0_dot * N_dot_0))
        
        return curr_mutual_inf
    
    def chi_squared(word_dict):
        N_11 = word_dict["N11"], N_10 = word_dict["N10"]
        N_01 = word_dict["N10"], N_00 = word_dict["N00"]
        N_1_dot = N_11 + N_10
        N_dot_1 = N_11 + N_01
        N_dot_0 = N_10 + N_00
        N_0_dot = N_01 + N_00

        current_chi_squared = \
            (N_11 + N_10 + N_01 + N_00) * ((N_11 * N_00 - N_10 * N_01)**2) / \
            ((N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00))
        return current_chi_squared
    
    for word in corpus1_dict:
        corpus1_dict[word]["MI"] = mutual_inf(word)
        corpus1_dict[word]["Chi_Sq"] = chi_squared(word)
    
    for word in corpus2_dict:
        corpus2_dict[word]["MI"] = mutual_inf(word)
        corpus2_dict[word]["Chi_Sq"] = chi_squared(word)




    





if __name__ == "__main__":
    main()
