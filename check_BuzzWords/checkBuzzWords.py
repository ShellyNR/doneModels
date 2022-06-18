import collections
import string

def checkBuzzWords(description):
    buzzWords = ["city", "living room", "rooms", "balcony", "parking", "tax", "street","neighborhood"] #insert as lower-case
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
            strMissing = "nature of the neighborhood"
            missingBuzz.remove("neighborhood")
            if len(missingBuzz) != 0:
                strMissing = strMissing + ", "
        strMissing = strMissing + ', '.join(missingBuzz)
        text = strMissing + ".\r\nIt will help you reach the relevant target audience."
    grade = ((buzzWordsCounter - missingBuzzCounter * 0.25) / buzzWordsCounter) * 100
    return grade, text

def checkLength(description):
    sentences = description.split(".")
    descriptionWordCount = len(description.split())
    counter = 0
    longSentenceNum = []
    for i, sentence in enumerate(sentences):
        wordsCounter = len(sentence.split())
        if 35 <= wordsCounter:
            counter += wordsCounter
            longSentenceNum.append(str(i+1))
    grade = ((descriptionWordCount - counter)/descriptionWordCount) * 100
    if len(longSentenceNum) != 0:
        if len(longSentenceNum) == 1:
            finalText = "You should shorten sentence #" + str(longSentenceNum[0]) + ".\r\n"
        else:
            finalText = "You should shorten those sentence - #" + ', #'.join(longSentenceNum) + ".\r\n"
        if len(longSentenceNum) != len(sentences):
            if len(sentences) == len(longSentenceNum)+1:
                finalText += "The other sentence is in great length."
            else:
                finalText += "The other sentences are in great length."
    else:
        finalText = "Your sentences are in great length!"
    return grade, finalText

def repeatedWords(description):
    repeatableWords = ["the", "and", "a", "that", "it", "not", "as", "there"] #insert as lower-case
    description = description.translate(str.maketrans('', '', string.punctuation)).lower()
    words = description.split()
    word_counts = collections.Counter(words)
    diffWords = len(word_counts)
    repeatWords = 0
    repeatingCounter = 0
    text = ""
    for word, count in sorted(word_counts.items()):
        if word not in repeatableWords and count >= 2:
            text = text + '"%s" is repeated %d times.\r\n' % (word, count)
            repeatingCounter += count
            repeatWords += 1
    if repeatWords != 0:
        text = text + "It is recommended to avoid repeating words in the description."
    else:
        text = text + "Excellent! Your description does not contain repeating words."
    grade = (((diffWords - repeatWords)/diffWords) * 100)
    if grade > repeatingCounter*5:
        grade = grade - (repeatingCounter*5)
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
            # print("We recommend you to change the word " + key + " to " + betterWordsConvertor[key])
            wrongWordsCounter += 1
            convertedDescription = convertedDescription.replace(key, betterWordsConvertor[key])
    text = ""
    if description != convertedDescription:
        text = convertedDescription
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
