import language_tool_python

def get_mistakes() :
    # print("hey: ", text)
    return tool.check(text)

def get_grade() :
    words = len(text.split(" "))
    mistakes_num = len(get_mistakes())
    return round((1 - (mistakes_num / words)) * 100)

def get_main_response() :
    grade = get_grade()
    if grade == 100 :
        return "No grammar mistakes were found!"
    if grade > 80 :
        return "Please notice minor grammar corrections"
    if grade > 50 :
        return "Please notice several grammar corrections"
    else :
        return "Pay attention to multiple grammar corrections"

def get_replacement_sentence() :
    mistakes_num = tool.check(text)
    return language_tool_python.utils.correct(text, mistakes_num)

def initialize(description):
    global tool
    global text
    tool = language_tool_python.LanguageTool('en-US')
    text = description

def get_issues() :
    issues_array = []
    issues_set = set()
    mistakes = get_mistakes()
    for mistake in mistakes :
        issues_set.add(mistake.message)
    issues_array = list(issues_set)
    return issues_array

def text_model(description) :
    initialize(description)
    response = {"issues": get_issues(), "main_response": get_main_response(), "grade": get_grade(),
                "replacement_description": get_replacement_sentence()}
    return response