# from monk.gluon_prototype import prototype
# import os
# import string
#
# gtf = prototype(verbose=1);
# gtf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True);

def renderList(list):
  list = set(list)
  if "Interior" in list:
    list.remove("Interior")
  if "living_room" in list:
    list.remove("living_room")
    list.add("living room")
  returnList = [x.lower() for x in list]
  return returnList

def predictAllPhotos():
  # photosNameList = os.listdir('/content/workspace/test')
  predictionsList = []
  #
  # for photoName in photosNameList:
  #   photoPath = "workspace/test/" + photoName
  #   predictions = gtf.Infer(img_name=photoPath);
  #   predictionsList.append(predictions["predicted_class"])
  #
  # return renderList(predictionsList)

def typeIsInObj(type, obj):
  types = type.split(",")
  for t in types:
    if t in obj:
      return True
  return False

def buildResponse(missingInDescription, missingInPhotos):
  response = ""
  if len(missingInDescription) != 0:
    response = "We saw that you attached a photo of: " + ', '.join(missingInDescription) + " - but you didn't mentioned it in the description.\r\n"
  if len(missingInPhotos) != 0:
    response = response + "We saw that in the description you mentioned: " + ', '.join(missingInPhotos) + " - but we didn't see matching image attached.\r\n"
  if response != "":
    response = response + "We recommend you to complete this information."
  return response

def roomType_model():
  print("INSIDEEEEEEEEEEEEEEEEEE")
  importantRoomType = ["bedroom", "bathroom,toilet,shower", "kitchen,cuisine", "living room,lounge,salon", "exterior,building,house,apartment"]
  description = "I love my bedroom it's nice"
  description = description.lower().translate(str.maketrans('', '', string.punctuation))
  predictionsList = predictAllPhotos()
  missingInDescription = []
  missingInPhotos = []

  for type in importantRoomType:
    firstType = type.split(",")[0]
    isInDescription = typeIsInObj(type, description)
    isInPredictionsList = typeIsInObj(type, predictionsList)
    if isInDescription and not isInPredictionsList: # is in description but not in images
      missingInPhotos.append(firstType)
    elif isInPredictionsList and not isInDescription: # is in images but not in description
      missingInDescription.append(firstType)

  response = buildResponse(missingInDescription, missingInPhotos)

  print(response)
  return response