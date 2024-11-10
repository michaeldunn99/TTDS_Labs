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


# def convert_non_alphanumeric_to_space(text_input):
#     """
#     Replace all non-alphanumeric and non-spacecharacters from the given text input with a space.
#
#     Args:
#         text_input (str): The input string from which non-alphanumeric characters are to be replaced.
#
#     Returns:
#         str: The resulting string after replacing all non-alphanumeric characters with space characters.
#     """
#     token_pattern = r"[^a-z\s]"
#     return  re.sub(token_pattern," ", text_input)


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

    if space_only(text_line):
        return ""
    
    processed_text = text_line.lower()
    
    if remove_stop_words:
        processed_text = processed_text.split()
        processed_text = remove_stop_words_function(processed_text, stop_words)
        processed_text = " ".join(processed_text)
    
    processed_text = convert_non_alphanumeric_to_space(processed_text)

    processed_text = remove_single_char(processed_text)
    
    processed_text = remove_end_single_char(processed_text)
    
    if apply_stemming:
        processed_text = processed_text.split()
        processed_text = my_porter_stemmer(processed_text, porter_stemmer)
        processed_text = " ".join(processed_text)


def main():
    with open('data/corpus1.txt') as corpus1:


if __name__ == "__main__":
    main()
