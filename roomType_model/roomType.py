from monk.gluon_prototype import prototype
import os
import string
import sys
from glob import glob

def buildResponse(missingInDescription, missingInPhotos):
    response = ""
    if len(missingInDescription) != 0:
        response = "We saw that you attached a photo of: " + ', '.join(
            missingInDescription) + " - but you didn't mention it in the description.\r\n"
    if len(missingInPhotos) != 0:
        response = response + "We saw that in the description you mentioned: " + ', '.join(
            missingInPhotos) + " - but we didn't see matching image attached.\r\n"
    if response != "":
        response = response + "We recommend you to complete this information."
    else:
        response = "Great job! There is no mismatching between your room type that mentioned in the text and in the photos."
    return response

def typeIsInObj(types, obj):
  for t in types:
    if t in obj:
      return True
  return False

def renderList(list):
    list = set(list)
    if "Interior" in list:
        list.remove("Interior")
    return [x.lower() for x in list]

def predictAllPhotos():
  ptf = prototype(verbose=1);
  ptf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True)
  photoPathList = glob(os.path.join('images', '*'))
  predictionsList = []
  for photoPath in photoPathList:
    predictions = ptf.Infer(img_name=photoPath)
    predictionsList.append(predictions["predicted_class"])
  return renderList(predictionsList)

def roomType_model(description):
    original_stdout = sys.stdout
    with open('output_roomType.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        importantRoomType = ["bedroom,bedrooms", "bathroom,bathrooms,toilet,shower", "kitchen,cuisine", "living room,living_room,livingroom,living-room,lounge,salon",
                             "building,exterior,house,apartment"]
        description = description.lower().translate(str.maketrans('', '', string.punctuation))
        missingInDescription = []
        missingInPhotos = []
        predictionsList = predictAllPhotos()

        for type in importantRoomType:
          types = type.split(",")
          firstType = types[0]

          isInDescription = typeIsInObj(types, description)
          isInPredictionsList = typeIsInObj(types, predictionsList)

          if isInDescription and not isInPredictionsList: # in description but not in images
            missingInPhotos.append(firstType)
          elif isInPredictionsList and not isInDescription: # in images but not in description
            missingInDescription.append(firstType)

        response = buildResponse(missingInDescription, missingInPhotos)
        sys.stdout = original_stdout
    os.remove('output_roomType.txt')
    return response
