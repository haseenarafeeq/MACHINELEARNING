
def replaceCommonWords(input_string):
    words_to_replace = {'a', 'about', 'an', 'and', 'any', 'are', 'as', 'at', 'by', 'can', 'do', 'for', 'from', 'get', 'have', 'how', 'in', 'is', 'may', 'me', 'my', 'need', 'of', 'on', 'or', 'please', 'some', 'the', 'this', 'that', 'there', 'to', 'what', 'What\'s', 'where', 'you'}
    # words_to_replace = {' ' + word + ' ' for word in words_to_replace}

    words = input_string.split()

    # Create a set of words to replace for faster lookups
    words_to_replace_set = set(words_to_replace)

    # Replace the specified words with a space
    replaced_words = [word for word in words if word not in words_to_replace_set]
    # Join the words back into a string
    result_string = ' '.join(replaced_words)

    return result_string