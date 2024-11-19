import unittest
import re
from mystery_corpora import remove_stop_words_function, default_stop_words
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

class TestRemoveStopWordsFunction(unittest.TestCase):

    def test_remove_stop_words(self):
        text_words = ["this", "is", "a", "test"]
        stop_words = default_stop_words
        expected_result = ["test"]
        result = remove_stop_words_function(text_words, stop_words)
        self.assertEqual(result, expected_result)

    def test_remove_stop_words_with_default(self):
        text_words = ["this", "is", "a", "test"]
        expected_result = ["test"]
        result = remove_stop_words_function(text_words)
        self.assertEqual(result, expected_result)

    def test_no_stop_words(self):
        text_words = ["this", "is", "her", "test"]
        stop_words = default_stop_words
        expected_result = ["test"]
        result = remove_stop_words_function(text_words, stop_words)
        self.assertEqual(result, expected_result)

    def test_all_stop_words(self):
        text_words = ["this", "is", "a", "test"]
        stop_words = default_stop_words
        expected_result = ["test"]
        result = remove_stop_words_function(text_words, stop_words)
        self.assertEqual(result, expected_result)

    def test_empty_text_words(self):
        text_words = []
        stop_words = {"this", "is", "a", "test"}
        expected_result = []
        result = remove_stop_words_function(text_words, stop_words)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()