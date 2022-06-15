import collections
import string

def checkBuzzWords(description):
    buzzWords = ["city", "living", "room", "balcony", "parking", "tax", "street","neighborhood"] #insert as lower-case
    buzzWordsCounter = len(buzzWords)
    missingBuzz = []
    description = description.translate(str.maketrans('', '', string.punctuation)).lower()
    words = description.split()
    for buzz in buzzWords:
        if buzz not in words:
            missingBuzz.append(buzz)
    missingBuzzCounter = len(missingBuzz)
    text = ""
    if missingBuzzCounter != 0:
        strMissing = ""
        if "neighborhood" in missingBuzz:
            strMissing = "nature of the neighborhood "
            missingBuzz.remove("neighborhood")
            if len(missingBuzz) != 0:
                strMissing = strMissing + ", "
        strMissing = strMissing + ', '.join(missingBuzz)
        text = "We recommend you to had details to your add about: " + strMissing + " - it's very important information for most peoples."
    grade = ((buzzWordsCounter - missingBuzzCounter * 0.25) / buzzWordsCounter) * 100
    return grade, text

def checkLength(description):
    sentences = description.split(".")
    descriptionWordCount = len(description.split())
    counter = 0
    text = ""
    for sentence in sentences:
        wordsCounter = len(sentence.split())
        if 30 <= wordsCounter:
            text = text + "you shold short this sentence : ' " + sentence + " '.\n"
            counter += wordsCounter
    grade = (counter/descriptionWordCount) * 50
    return grade, text

def repeatedWords(description):
    repeatableWords = ["the", "and", "a", "that", "it", "not", "as", "there"] #insert as lower-case
    description = description.lower()
    words = description.split()
    word_counts = collections.Counter(words)
    diffWords = len(word_counts)
    repeatWords = 0
    repeatingCounter = 0
    text = ""
    for word, count in sorted(word_counts.items()):
        if word not in repeatableWords and count >= 2:
            text  = text + '"%s" is repeated %d times - please do not repeat it.\n' % (word, count)
            repeatingCounter += count
            repeatWords += 1
    grade = (((diffWords - repeatWords)/diffWords) * 100) - (repeatingCounter*5)
    return grade, text

def checkGoodDescriptionWords(description):
    betterWordsConvertor = {"nice":"beautiful", "old":"vintage"} #insert as lower-case
    betterKeys = betterWordsConvertor.keys()
    description = description.lower()
    words = description.split()
    words = set(words)
    wrongWordsCounter = 0
    convertedDescription = description
    for key in betterKeys:
        if key in words:
            print("we recommand you to change the word " + key + " to " + betterWordsConvertor[key])
            wrongWordsCounter += 1
            convertedDescription = convertedDescription.replace(key, betterWordsConvertor[key])
    text = ""
    if description != convertedDescription:
        text = "The better way to write your description is: " + convertedDescription
    grade = 100 - wrongWordsCounter * 5
    return grade, text

def check_text_quality(description):
    lengthGrade, lengthText = checkLength(description)
    repeatedWordsGrade, repeatedWordsText = repeatedWords(description)
    BuzzWordsGrade, BuzzWordsText = checkBuzzWords(description)
    goodDescriptionWordsGrade, goodDescriptionWordsText = checkGoodDescriptionWords(description)
    response = {"length_Grade": lengthGrade, "length_text": lengthText , "repeated_Words_Grade": repeatedWordsGrade,
                "repeated_text": repeatedWordsText, "BuzzWords_Grade": BuzzWordsGrade, "BuzzWords_text": BuzzWordsText,
                "good_description_words_Grade": goodDescriptionWordsGrade, "good_description_words_text": goodDescriptionWordsText}
    return response