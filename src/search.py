import sys
import os
import xml.etree.ElementTree as ET
import re
import numpy as np
from collections import defaultdict, OrderedDict
import Stemmer
import time

#################################################################################################################################
#   Notes on EXECUTION:
#
#       1.  To run on command line, navigate to directory containing .py script and xml document to be read into index. Type: 
#           
#               code.py trec.5000.xml 
#       
#           where trec.5000.xml is a command line argument
#
#       2.  The system will then prompt the user to state whether they would like to implement (i) Stopping (ii) Stemming
#           - The user must respond "Y" or 'N" to both questions (depending on preference)
#       
#       3.  The system will then run and output the three files (i) index.txt (ii) results.boolean.txt (iii) results.ranked.txt
#           to the current directory.
#
#################################################################################################################################
#
#   Notes on CODE STRUCTURE
#       
#   The code below is structured in three sections as follows:
#
#       1. Auxiliary Functions
#           -   Standalone functions involved in the implementation but do not directly use the inverted index dictionary 
#               thus judged to be separate concerns and so separated
#           -   These functions are called by the methods in the Inverted Index class
#           -   Include functions for:
#                   - Preprocessing text & XML
#                   - Parsing Boolean and Ranked IR queries from file
#                   - Calculating proximity between items in two separate distinct ordered lists
#                   - Ensuring command line arguments passed correctly
#       
#       2. InvertedIndex Class
#           - Implementation of the InvertedIndex class which includes:
#                   - Instantation method (building inverted index data structure)
#                   - Search methods: Phrase search, Boolean search, Proximity Search, Ranked_IR_TFIDF
#           - Also includes functions to output inverted index data structure to file
#
#
#       3. Main Function:
#           - Controls the overall systems execution (i.e function/method calling) and IO operations:
#           - Calls functions from Section 1 and 2 in order to:
#                   - Ensure command line arguments entered correctly (calling function from section 1)
#                   - Obtain input from user with regards to applying stopping and stemming
#                   - Instantiate an InvertedIndex object
#                   - Call method that writes in index data structure to file
#                   - Reads lists of Boolean and Ranked IR queries into memory, calls functions to parse them
#                     then calls search methods of InvertedIndex to apply search, then writes output to file
#
#        


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#
#       Section 1: AUXILIARY FUNCTIONS
#
#           Subsection A: Set of Stopping Words and Stemming
#       


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


#################################################################################################################################
#
#       SECTION 1: AUXILIARY FUNCTIONS
#
#           Subsection B: Preprocessing functions
#

def space_only(line):
    pattern = r"^[\s\n]+$"
    return bool(re.match(pattern, line))

def convert_non_alphanumeric_to_space(text_input):
    """
    Replace all non-alphanumeric and non-spacecharacters from the given text input with a space.

    Args:
        text_input (str): The input string from which non-alphanumeric characters are to be replaced.

    Returns:
        str: The resulting string after replacing all non-alphanumeric characters with space characters.
    """
    token_pattern = r"[^\w\s]"
    return  re.sub(token_pattern," ", text_input)

def remove_single_s(text_input):
    """
    Removes instances of the letter 's' surrounded by spaces from the input text.

    Args:
        text_input (str): The input string from which to remove ' s '.

    Returns:
        str: The modified string with ' s ' removed.
    """
    pattern = r" s "
    return  re.sub(pattern," ", text_input)

def remove_end_single_s(text_input):
    """
    Remove the singular letter ' s' at the end of lines in the given text input.

    Args:
        text_input (str): The input text where ' s' at the end of lines should be removed.

    Returns:
        str: The modified text with ' s' removed from the end of lines.
    """
    pattern = r" s\n"
    return  re.sub(pattern,"\n", text_input)


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

    processed_text = remove_single_s(processed_text)
    
    processed_text = remove_end_single_s(processed_text)
    
    if apply_stemming:
        processed_text = processed_text.split()
        processed_text = my_porter_stemmer(processed_text, porter_stemmer)
        processed_text = " ".join(processed_text)
    
    if space_only(processed_text):
        return ""
 
    return processed_text


#################################################################################################################################
#
#       SECTION 1: AUXILIARY FUNCTIONS
#
#           Subsection C: Parsing functions



#Ensure that the user runs the program with a XML file as command line argument to be read into Inverted Index
def check_command_line_arguments(command_line_arguments: list, file_type: str):
    """
    Checks the command line arguments to ensure that at least one file is provided 
    and that all files match the specified file type.

    Args:
        command_line_arguments (list): List of command line arguments.
        file_type (str): The expected file type (e.g., "xml", "txt").

    Raises:
        SystemExit: If no files are provided or if any file does not match the specified file type.
    """
    if len(command_line_arguments) < 2:
        sys.exit("Must enter one or more text files to be preprocessed")
    
    pattern = rf"^.*\.{file_type}$"
    
    for file in command_line_arguments[1:]:
        if not re.search(pattern, file):
            sys.exit("Please only enter alphanumeric files with the specified extension as command line arguments")


def parse_xml(file_path):
    """
    Parses an XML file and returns the root element.

    Args:
        file_path (str): The path to the XML file to be parsed.

    Returns:
        xml.etree.ElementTree.Element: The root element of the parsed XML tree.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root


def preprocess_xml(file_path, remove_stop_words, apply_stemming):
    """
    Preprocesses an XML file by parsing it, extracting document elements, and applying a preprocessing function
    to the text within 'Headline' and 'TEXT' elements.
    Args:
        file_path (str): The path to the XML file to be processed.
    Returns:
        ElementTree.Element: The root element of the processed XML tree.
    """
    my_root = parse_xml(file_path)

    for doc in my_root.findall('DOC'):
        doc_id = doc.find('DOCNO').text.strip()
        
        headline_element = doc.find('HEADLINE')
        if headline_element is not None:
            processed_headline = my_preprocessor(headline_element.text.strip(), remove_stop_words, apply_stemming)
            headline_element.text = processed_headline
        
        text_element = doc.find('TEXT')
        if text_element is not None:
            current_text = text_element.text.strip()
            processed_text = my_preprocessor(current_text, remove_stop_words, apply_stemming)
            text_element.text = processed_text
    return my_root

def parse_question_file(line_of_text):
    """
    Parses a line of text to remove question identifiers.

    This function uses a regular expression to identify and remove
    question identifiers from the beginning of a line of text. The
    question identifiers are expected to be in the format of an optional
    'q' followed by one or two digits, an optional colon, and one or more
    whitespace characters.

    Args:
        line_of_text (str): The line of text to be parsed.

    Returns:
        str: The line of text with the question identifier removed.
    """
    question_regex_pattern = r"^q?[0-9][0-9]?:?(\s)+"
    return re.sub(question_regex_pattern, "", line_of_text)

def boolean_query_parser(documents_function, not_function, text):
    """
    Parses a boolean query string and generates a list of document sets and connectors.
    Args:
        documents_function (function): A function that takes a text phrase and returns a set of documents containing that phrase.
        not_function (function): A function that takes a set of documents and returns the inverse set.
        text (str): The boolean query string in the format '(NOT) phrase (AND|OR) (NOT) phrase (AND|OR) (NOT) phrase' etc.
    Returns:
        list: A list containing document sets and connectors (AND/OR) in the order they appear in the query string.
    Raises:
        SystemExit: If the query string is not in the correct format.
    """
    query_regex_pattern = r"^(?P<initial>(NOT)?((?!\bAND\b|\bOR\b).)+)(?P<Connector>(\bAND\b|\bOR\b)?)(?P<Remaining_Text>.*)$"
    NOT_regex_pattern = r"^(?P<NOT_group>(NOT)?)(?P<Text>((?!\bAND\b|\bOR\b).)+$)"
    document_AND_OR_list = []

    continue_loop = True
    while continue_loop == True:
        match = re.match(query_regex_pattern, text)

        if not match:
            print("Enter in the format '(NOT) phrase (AND|OR) (NOT) phrase (AND|OR) (NOT) phrase' etc")
            sys.exit(-1)
        else:
            initial_group = match.group("initial")
            initial_phrase_match = re.match(NOT_regex_pattern, initial_group)
            initial_NOT = initial_phrase_match.group("NOT_group").strip()
            initial_text = initial_phrase_match.group("Text").strip()
            document_set = documents_function(initial_text)
            if not initial_NOT:
                #generate document set for the initial text
                #add document set to list
                document_AND_OR_list.append(document_set)
            else:
                #generate inverse document set for the initial text
                #add document set to list
                not_function_output = not_function(document_set)
                document_AND_OR_list.append(not_function_output)
            connector = match.group("Connector")
            remaining_text = match.group("Remaining_Text").strip()
        
        if (not connector) and (not remaining_text):
            continue_loop = False

        elif (not connector) or (not remaining_text):
            print("Enter in the format '(NOT) phrase (AND|OR) (NOT) phrase (AND|OR) (NOT) phrase' etc")
            sys.exit(-1)
        
        else:
            document_AND_OR_list.append(connector)
                #Add conector to list
            text = remaining_text
    
    return document_AND_OR_list

def parse_proximity_query(proximity_query_text):
    """
    Parses a proximity query string and extracts the phrases and proximity value.

    The proximity query string should be in the format:
    `#<Prox>(<Phrase_1>,<Phrase_2>)`
    where `<Prox>` is a positive integer representing the proximity value,
    and `<Phrase_1>` and `<Phrase_2>` are the phrases to be matched.

    Args:
        proximity_query_text (str): The proximity query string to be parsed.

    Returns:
        tuple: A tuple containing:
            - A tuple of two strings: (phrase_1, phrase_2)
            - An integer representing the proximity value

    Raises:
        SystemExit: If the input string does not match the expected format.
    """
    parsed_query = re.match(r"^\#(?P<Prox>\d\d*)\((?P<Phrase_1>[\w\s]+),(?P<Phrase_2>[\w\s]+)\)$", proximity_query_text)
    if not parsed_query:
        print("Not a match")
        sys.exit(-1)
    phrase_1 = parsed_query.group("Phrase_1")
    phrase_2 = parsed_query.group("Phrase_2")
    prox = int(parsed_query.group("Prox"))
    return (phrase_1, phrase_2), prox

#################################################################################################################################
#
#       SECTION 1: AUXILIARY FUNCTIONS
#
#           Subsection D: Proximity function

def proximity(list_0, list_1):
    """
    Calculate the minimum proximity between elements from two sorted lists containing distinct elements.
    This function merges two sorted lists while keeping track of the minimum proximity (absolute difference)
    between consecutive elements from different lists. It returns the minimum proximity and the elements
    of the elements that have this minimum proximity.
    Parameters:
    list_0 (list): The first sorted list of numerical values.
    list_1 (list): The second sorted list of numerical values.
    Returns:
    tuple: A tuple containing:
        -   min_proximity (float): 
                The minimum proximity (absolute difference) between elements from the two lists.
        
        -   positions (list of tuples): 
                A list of tuples where each tuple contains the positions of the elements
                from the two lists that have the minimum proximity.
                    NOTE: In the tuple, the first argument is the element from list_0 and 
                the second argument is the position from list_1.
    """
    min_proximity = np.inf
    positions = []
    min_prox_list_0_pos = np.inf
    min_prox_list_1_pos = np.inf
    list_0_counter = 0
    list_1_counter = 0
    i = 1
    previous_list_add = 1000
    merged_list = []

    #Edge case: Starting
    if list_0[list_0_counter] <= list_1[list_1_counter]:
        merged_list.append(list_0[list_0_counter])
        list_0_counter +=1
        previous_list_add = 0
    else:
        merged_list.append(list_1[list_1_counter])
        list_1_counter +=1
        previous_list_add = 1

    #Remainder of the list
    while i < (len(list_0) + len(list_1)):
        #If we have already added all of list 0 then add the next one from list 1 and check proximity
        #Then add the rest of list 0 to the list and we are finished
        
        #Special case: If we have already added all of list 0 then add the next one from list 1 and check proximity
        if list_0_counter == len(list_0):
            current_list_add = 1
            merged_list.append(list_1[list_1_counter])
            list_1_counter +=1
            if current_list_add + previous_list_add == 1:
                current_proximity = np.abs(merged_list[i]- merged_list[i-1])
                if current_proximity < min_proximity:
                    min_proximity = current_proximity
                    if current_list_add == 1:
                        min_prox_list_1_pos = merged_list[i]
                        min_prox_list_0_pos = merged_list[i-1]
                        
                    else:
                        min_prox_list_0_pos = merged_list[i]
                        min_prox_list_1_pos = merged_list[i-1]
                    positions = [(min_prox_list_0_pos, min_prox_list_1_pos)]
             
                elif current_proximity == min_proximity:
                    if current_list_add == 1:
                        positions.append((merged_list[i-1],merged_list[i]))
                    else:
                        positions.append((merged_list[i],merged_list[i-1]))
            if list_1_counter < len(list_1):
                merged_list.append(list_1[list_1_counter:])
            i = len(list_0) + len(list_1)

        #Else if we have already added all of list 1 then add the next one from list 0 and check proximity
        #Then add the rest of list 0 to the list and we are finished
        elif list_1_counter == len(list_1):
            current_list_add = 0
            merged_list.append(list_0[list_0_counter])
            list_0_counter += 1
            if current_list_add + previous_list_add == 1:
                current_proximity = np.abs(merged_list[i] - merged_list[i-1])
                if current_proximity < min_proximity:
                    min_proximity = current_proximity
                    if current_list_add == 1:
                        min_prox_list_1_pos = merged_list[i]
                        min_prox_list_0_pos = merged_list[i-1]
                    else:
                        min_prox_list_0_pos = merged_list[i]
                        min_prox_list_1_pos = merged_list[i-1]
                    positions = [(min_prox_list_0_pos, min_prox_list_1_pos)]
                elif current_proximity == min_proximity:
                    if current_list_add == 1:
                        positions.append((merged_list[i-1],merged_list[i]))
                    else:
                        positions.append((merged_list[i],merged_list[i-1]))
            if list_0_counter < len(list_0):
                merged_list.append(list_0[list_0_counter:])
            i = len(list_0) + len(list_1)
            break
            
            
        #Regular case: If the current item to be added is from list 0 then add it and check proximity
        elif list_0[list_0_counter] <= list_1[list_1_counter]:
            current_list_add = 0
            merged_list.append(list_0[list_0_counter])
            list_0_counter +=1
            if current_list_add + previous_list_add == 1:
                current_proximity = np.abs(merged_list[i]- merged_list[i-1])
                if current_proximity < min_proximity:
                    min_proximity = current_proximity
                    if current_list_add == 1:
                        min_prox_list_1_pos = merged_list[i]
                        min_prox_list_0_pos = merged_list[i-1]
                    else:
                        min_prox_list_0_pos = merged_list[i]
                        min_prox_list_1_pos = merged_list[i-1]
                    positions = [(min_prox_list_0_pos, min_prox_list_1_pos)]
                elif current_proximity == min_proximity:
                    if current_list_add == 1:
                        positions.append((merged_list[i-1],merged_list[i]))
                    else:
                        positions.append((merged_list[i],merged_list[i-1]))
            previous_list_add = current_list_add
        #Other regular case: If the current item to be added is from list 1 then add it and check proximity
        else:
            current_list_add = 1
            merged_list.append(list_1[list_1_counter])
            list_1_counter +=1
            if current_list_add + previous_list_add == 1:
                current_proximity = np.abs(merged_list[i]- merged_list[i-1])
                if current_proximity < min_proximity:
                    min_proximity = current_proximity
                    if current_list_add == 1:
                        min_prox_list_1_pos = merged_list[i]
                        min_prox_list_0_pos = merged_list[i-1]
                    else:
                        min_prox_list_0_pos = merged_list[i]
                        min_prox_list_1_pos = merged_list[i-1]
                    positions = [(min_prox_list_0_pos, min_prox_list_1_pos)]
                elif current_proximity == min_proximity:
                    if current_list_add == 1:
                        positions.append((merged_list[i-1],merged_list[i]))
                    else:
                        positions.append((merged_list[i],merged_list[i-1]))
            previous_list_add = current_list_add
        i+=1

        #If the previous item to be added is from a different list to the current one then consider
        #if we have a new minimum proximity
    return min_proximity, positions


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#
#     Section 2: Inverted Index Class
#

class InvertedIndex:
    def __init__(self, file_path, remove_stop_words, apply_stemming):
        """
        Initializes the class with the given parameters and builds the inverted index.

        Args:
            file_path (str): The path to the XML file to be processed.
            remove_stop_words (bool): Flag indicating whether to remove stop words during preprocessing.
            apply_stemming (bool): Flag indicating whether to apply stemming during preprocessing.

        Attributes:
            _root (Element): The root element of the preprocessed XML.
            _inverted_index (defaultdict): The inverted index storing term frequencies, document sets, and position dictionaries.
            _apply_stemming (bool): Indicates if stemming is applied.
            _remove_stop_words (bool): Indicates if stop words are removed.
        """
        self._root = preprocess_xml(file_path, remove_stop_words, apply_stemming)
        self._inverted_index = defaultdict(lambda: {"frequency": 0, "document_set": set(), "position_dict": defaultdict(lambda:set())})
        #Assign a special key to the inverted index that contains the set of all documents and the total number of documents
        self._inverted_index["_total_document_set"] = {"size":0, "document_set": set()}
        self._apply_stemming = apply_stemming
        self._remove_stop_words = remove_stop_words
        self._build_index()
        
    def _build_index(self):
        """
        Builds an inverted index from the XML documents found in the root element.

        The method processes each document (DOC) in the root, extracting the document ID (DOCNO),
        headline (HEADLINE), and text (TEXT). It updates the inverted index with the frequency,
        document set, and position of each word found in the headline and text.

        The inverted index structure is as follows:
        - word: {
            "frequency": int,
            "document_set": set of document IDs,
            "position_dict": {
            document_id: set of positions
            }
        }
        - "_total_document_set": {
            "document_set": set of all document IDs,
            "size": int
        }

        The method also updates the total document set with all document IDs and calculates its size.

        Note:
        - The headline words' positions are indexed starting from 0.
        - The text words' positions are indexed starting from the length of the headline.

        Raises:
            AttributeError: If the XML structure does not match the expected format.
        """
        for doc in self._root.findall('DOC'):
            headline_length = 0
            doc_id = int(doc.find('DOCNO').text.strip())
            headline_element = doc.find('HEADLINE')
            if headline_element is not None:
                headline_enumerable = enumerate(headline_element.text.split())
                headline_list = list(headline_enumerable)
                headline_length = headline_list[-1][0] + 1 if headline_list else 0
                for i, word in headline_list:
                    self._inverted_index[word]["frequency"] += 1
                    self._inverted_index[word]["document_set"].add(doc_id)
                    self._inverted_index[word]["position_dict"][doc_id].add(i)
                    self._inverted_index["_total_document_set"]["document_set"].add(doc_id)
            text_element = doc.find('TEXT')
            if text_element is not None:
                text_enumerable = enumerate(text_element.text.split())
                for i, word in text_enumerable:
                    self._inverted_index[word]["frequency"] += 1
                    self._inverted_index[word]["document_set"].add(doc_id)
                    self._inverted_index[word]["position_dict"][doc_id].add(i + headline_length)
                    self._inverted_index["_total_document_set"]["document_set"].add(doc_id)
        self._inverted_index["_total_document_set"]["size"] = len(self._inverted_index["_total_document_set"]["document_set"])
    
    def write_index_to_file(self, file_path):
        """
        Writes the inverted index to a specified file.

        The method writes each word in the inverted index to the file, along with its frequency and the document IDs 
        where the word appears. For each document ID, it also writes the positions of the word within the document.

        Args:
            file_path (str): The path to the file where the inverted index will be written.

        Raises:
            IOError: If the file cannot be opened or written to.
        """
        with open(file_path, 'w') as output_file:
            for word in (key for key in self._inverted_index.keys() if key != "_total_document_set"):
                    output_file.write(f"{word}:{self._inverted_index[word]['frequency']}\n")
                    #output_file.write(f"\tFrequency: {self._inverted_index[word]['frequency']}\n")
                    for doc_id in sorted(list(self._inverted_index[word]["document_set"])):
                        output_file.write(f"\t{doc_id}: ")
                        term_list = sorted(list(self._inverted_index[word]["position_dict"][doc_id]))
                        term_frequency = len(term_list)
                        for i in range(term_frequency-1):
                            output_file.write(f"{term_list[i]},")
                        output_file.write(f"{term_list[term_frequency-1]}\n")

    def _word_dict(self, word):
        """
        Checks if a word exists in the inverted index and returns its dictionary.

        Args:
            word (str): The word to check in the inverted index.

        Returns:
            tuple: A tuple containing a boolean and a dictionary. The boolean is True if the word's frequency is greater than 0,
                   indicating the word is in the document, and False otherwise. The dictionary contains the word's details from
                   the inverted index.
        """
        word_dict = self._inverted_index[word]
        #Since _inverted_index is a defaultdict, we can just check if the word is a key without worrying about key errors
        # since the defaultdict will create a new key value pair if the key doesn't exist,  where "frequency" has value 0
        # We return False if the frequency is 0 (i.e. the word is not in the document) and True if it is, 
        # also returning the dictionary
        if word_dict["frequency"]:
            return True, word_dict
        else:
            return False, word_dict
        
    def _phrase_search(self, text_input_string):
        """
        Perform a phrase search on the given text input string.
        This method processes the input string, checks for the presence of each word in the phrase,
        and verifies if the words appear consecutively in the documents.
        Args:
            text_input_string (str): The input string containing the phrase to search for.
        Returns:
            tuple: A tuple containing a boolean and a dictionary.
                - The boolean indicates whether the phrase was found (True) or not (False).
                - The dictionary contains the documents and their respective positions where the phrase starts.
                  If the phrase is not found, an empty dictionary is returned.
        """
        word_list = my_preprocessor(text_input_string, remove_stop_words=self._remove_stop_words, apply_stemming=self._apply_stemming).split()
        #First loop through the word list to check each of our words are actually there
        #before we start to check ifthe they are there consecutively (as a phrase)
        word_dicts = []
        for word in word_list:
            word_present_boolean, word_dict = self._word_dict(word)
            if not word_present_boolean:
                return False, {}
            else:
                word_dicts.append(word_dict)
        document_set = word_dicts[0]["document_set"]

        #We are going to maintain our first word position dict for finding phrases
        #potential_matching_position_dict = word_dicts[0]["position_dict"]
        potential_positions_dict = {document:word_dicts[0]["position_dict"][document].copy() for document in document_set}
        for i in range(len(word_list)-1):
            docs_with_no_matches = 0
            docs_to_remove = set()
            next_word_dict = word_dicts[i+1]
            current_word_dict = word_dicts[i]
            document_set = document_set & next_word_dict["document_set"]
            potential_positions_dict = {document:potential_positions_dict[document] for document in document_set}
            
            #Loop through each of the documents have all the currently searched words
            for match_doc in document_set:
                #For each of the positions that the current word exists in the current docs
                for position in current_word_dict["position_dict"][match_doc]:

                    #Check if the the next_word_dict has a position that is one after (i.e. the words are consecutive)
                    if (position+1) not in next_word_dict["position_dict"][match_doc]:
                        #Remove the position of the first word in the potential matching dict
                            if ((position - i) in potential_positions_dict[match_doc]):
                                potential_positions_dict[match_doc].remove(position - i)
                if len(potential_positions_dict[match_doc]) == 0:
                    docs_to_remove.add(match_doc)
                    docs_with_no_matches += 1
            
                if docs_with_no_matches == len(potential_positions_dict.keys()):
                    return False, {}
            
            for doc in docs_to_remove:
                document_set.remove(doc)
                del potential_positions_dict[doc]

        if not potential_positions_dict:
            return False, {}
        
        sorted_docs_first_position_keys = sorted([int(key) for key in potential_positions_dict.keys()])
            
        
        sorted_docs_first_positions_dict = {match_doc: sorted(list(potential_positions_dict[match_doc])) for match_doc in sorted_docs_first_position_keys}

        return True, sorted_docs_first_positions_dict
        
    def _documents_containing_phrase(self, phrase):
        """
        Determines the set of documents that contain the given phrase.

        Args:
            phrase (str): The phrase to search for within the documents.

        Returns:
            set: A set of document identifiers where the phrase is present. 
                 Returns an empty set if the phrase is not found in any document.
        """
        phrase_present, document_and_position_dict = self._phrase_search(phrase)
        #Since _inverted_index is a defaultdict, we can just check if the word is a key without worrying about key errors
        # since the defaultdict will create a new key value pair if the key doesn't exist,  where "frequency" has value 0
        # We return empty set if the frequency is 0 (i.e. the word is not in the document) and True if it is, 
        # also returning the dictionary
        if phrase_present:
            return set(document_and_position_dict.keys())
        else:
            #If the word is not in the document, we return an empty set
            return set()
        
    def _not_operator(self, document_set):
        total_docs_set = self._inverted_index["_total_document_set"]["document_set"]
        doc_set_difference = total_docs_set - document_set
        return doc_set_difference
    
    def _and_operator(self, list_of_document_sets):
        combined_document_set = set.intersection(*list_of_document_sets)
        return combined_document_set
        
    def _or_operator(self, list_of_document_sets):
        combined_document_set = set.union(*list_of_document_sets)
        return combined_document_set
        
    def _proximity_search(self, tuple_of_two_phrases, proximity_value):
        """
        Perform a proximity search for two phrases within a specified proximity value.
        This method checks if two given phrases appear within a certain proximity of each other
        in the documents. It returns a sorted list of document IDs where the phrases are found
        within the specified proximity.
        Args:
            tuple_of_two_phrases (tuple): A tuple containing two phrases to search for.
            proximity_value (int): The maximum allowed distance between the two phrases in the documents.
        Returns:
            list: A sorted list of document IDs where the phrases are found within the specified proximity.
                  If no such documents are found, an empty list is returned.
        """
        list_of_position_dicts = []
        for phrase in tuple_of_two_phrases:
            #Phrase search will tell us whether that phrase file in the document
            #It also returns a dictionary of all the documents and positions of those documents
            phrase_present, dict_of_phrase_locations = self._phrase_search(phrase)
            if phrase_present == False:
                return []
            else:
                list_of_position_dicts.append(dict_of_phrase_locations)

        
        phrase_1_position_dict = list_of_position_dicts[0]
        phrase_2_position_dict = list_of_position_dicts[1]
        #Now we have a list of length two, each consisting of a tuple (True, phrase_position_dict)
        #where phrase_position_dict is a dictionary with document numbers as keys and positions of that phrase
        #in the document as values

        #Now we need to check, for the documents (keys) that overlap, how close the phrases are (i.e. minimal distance between positions)

        #Docs that both phrases are in are the overlapping set of keys
        #Make a set that contains the documents that both phrases are in 
        common_docs = set(phrase_1_position_dict.keys()) & set(phrase_2_position_dict.keys())

        #Go through the documents both phrases are in and calculate the closest distance between those phrases in that document
        proximity_list = []
        for doc in common_docs:
            proximity_list.append((doc, proximity(phrase_1_position_dict[doc], phrase_2_position_dict[doc])))                                                             
        
        min_set = set()

        for doc_tuple in proximity_list:
            if doc_tuple[1][0] <= proximity_value:
                min_set.add(doc_tuple[0])

        if not min_set:
            return []
        else:
            return sorted(list(min_set))
        
    #Auxiliary function to the main ranked_ir_tfidf function below
    def _tfidf_term_weighting(self, term_dict, document):
        """
        Calculate the TF-IDF term weighting for a given term in a document.
        Args:
            term_dict (dict): A dictionary containing term information, including:
                - "position_dict" (dict): A dictionary where keys are document identifiers and values are lists of positions where the term appears in the document.
                - "document_set" (set): A set of documents in which the term appears.
            document (str): The identifier of the document for which to calculate the term weighting.
        Returns:
            float: The TF-IDF term weighting for the term in the specified document. Returns 0 if the term is not in the document.
        """

        #Check if the search term is in the document we are looking a
        if document not in term_dict["document_set"]:
            return 0
        
        my_N = self._inverted_index["_total_document_set"]["size"]

        term_frequency = len(term_dict["position_dict"][document])

        document_frequency = len(term_dict["document_set"])

        w_t_d = (1 + np.log10(term_frequency))*np.log10((my_N / document_frequency))

        return w_t_d
    
    def ranked_ir_tfidf(self, text_query):
        """
        Computes the TF-IDF weighted scores for a given text query and returns a sorted dictionary of documents with non-zero scores.

        Args:
            text_query (str): The input query text for which the TF-IDF scores need to be computed.

        Returns:
            dict: A dictionary where keys are document identifiers and values are their corresponding TF-IDF scores, sorted in descending order. Documents with a score of zero are excluded.
        """
        list_of_terms = my_preprocessor(text_query, self._remove_stop_words, self._apply_stemming).split()
        wtd_score_dict = defaultdict(lambda : 0)
        for term in list_of_terms:
            term_present_in_collection, term_dict = self._word_dict(term)
            if term_present_in_collection:
                document_set = term_dict["document_set"]
                for document in document_set:
                    wtd_score_dict[document] += self._tfidf_term_weighting(term_dict, document)
        sorted_wtd_score = OrderedDict(sorted(wtd_score_dict.items(), key = lambda x: x[1], reverse=True))
        remove_zero_sorted_wtd_score = {key:sorted_wtd_score[key] for key in sorted_wtd_score}
        return remove_zero_sorted_wtd_score


    def boolean_query(self, text):
        """
        Processes a boolean query and returns a sorted list of document IDs that match the query.
        The method parses the input text to identify AND and OR operators, then processes the query
        by first handling AND operations within OR groups, and finally combining the results using OR operations.
        Args:
            text (str): The boolean query string containing terms and operators (AND, OR).
        Returns:
            list: A sorted list of document IDs that match the boolean query.
        """
        _AND_OR_list = boolean_query_parser(self._documents_containing_phrase, self._not_operator, text)
        _AND_list = []
        OR_Positions = []
        _No_ORs = 0
        for i in range(len(_AND_OR_list)):
            if _AND_OR_list[i] == "OR":
                if _No_ORs  == 0:
                    _AND_list.append(_AND_OR_list[:i])
                else:
                    _AND_list.append(_AND_OR_list[OR_Positions[_No_ORs-1]+1:i])
                _No_ORs +=1
                OR_Positions.append(i)
        if _No_ORs > 0:
            _AND_list.append(_AND_OR_list[OR_Positions[_No_ORs-1]+1:])
        else:
            _AND_list.append(_AND_OR_list)
        
        OR_sets = []
        for list_of_sets in _AND_list:
            list_of_sets = [item for item in list_of_sets if item != "AND"]
            OR_sets.append(self._and_operator(list_of_sets))
        
        final_set = self._or_operator(OR_sets)

        final_set = sorted(list(final_set))

        return final_set

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#
#     Section 3: Main function

def main():

    #Ensure command line arguments entered correctly
    check_command_line_arguments(sys.argv, "xml")

    #Prompt user to state whether stopping should be applied
    remove_stop_words = input("Do you want to remove stop words? (Y/N): ")
    if remove_stop_words == "Y":
        remove_stop_words = True
    elif remove_stop_words == "N":
        remove_stop_words = False
    else:
        sys.exit(-1)
    
    #Prompt user to state whether stemming should be applied
    apply_stemming = input("Do you want to apply_stemming? (Y/N): ")
    if apply_stemming == "Y":
        apply_stemming = True
    elif apply_stemming == "N":
        apply_stemming = False
    else:
        sys.exit(-1)

    #Generate a word index
    my_collection_index = InvertedIndex(sys.argv[1], remove_stop_words, apply_stemming)

    #Write index to disc
    my_collection_index.write_index_to_file(f'output/index.txt')

    # Read in Boolean queries, parse them, apply Boolean Search, write results to disc
    with open("data/queries.lab2.txt", 'r') as file, open("output/results.boolean.txt", "w") as boolean_output:
        for i, line in enumerate(file):
            text = parse_question_file(line)
            if text[0] != "#":
                output_list = my_collection_index.boolean_query(text)
                if output_list:
                    for doc in output_list:
                        boolean_output.write(f"{i+1},{doc}\n")
            else:
                phrase_tuple, proximity = parse_proximity_query(text)
                output_list = my_collection_index._proximity_search(phrase_tuple, proximity)
                if output_list:
                    for doc in output_list:
                        boolean_output.write(f"{i+1},{doc}\n")
    
    # Read in ranked IR queries, parse them, apply Ranked IR search function, write results to disc
    with open("data/queries.lab3.txt",'r') as file, open("output/results.ranked.txt", "w") as ranked_ir_output:
        for i, line in enumerate(file):
            parsed_line = parse_question_file(line)
            ranked_ir_dict = my_collection_index.ranked_ir_tfidf(parsed_line)
            for document in ranked_ir_dict:
                    ranked_ir_output.write(f"{i+1},{document},{ranked_ir_dict[document]:.4f}\n")

    while True:
        #get user input
        user_input = input("Enter your query: ")
        if user_input == "exit":
            break
        else:
            #Apply boolean query
            output_list = my_collection_index.boolean_query(user_input)
            if output_list:
                print(output_list)
            else:
                print("No documents found")

    
 
if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")