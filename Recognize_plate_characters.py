import pickle
import tkinter as tk
from tkinter import *
import License_setup
from sklearn.metrics import confusion_matrix
import UI
print("LOADING MODEL...")
root1 = Tk()
#root.title("anpr")
file1=""

frame1 = tk.Frame(root1, bg='grey')
modelPath = "./dataModelNEWFINAL.sav"
loadModel = pickle.load(open(modelPath,"rb"))

print("#################################################")
print("MODEL LOADED")
print(loadModel)

resultPredictTop =[]

########################################## TOP HALF OF PLATE LETTERS PREDICTION ########################################

for each_top_Letter in reversed(License_setup.charactersTop):

    each_top_Letter = each_top_Letter.reshape(1,-1)
    result_top = loadModel.predict(each_top_Letter)
    resultPredictTop.append(result_top)

#PRINTING PLATE DATA

plate_characters_top = ""
print("..................................................")
print("...Predicting top half plate characters... ")
for each_top_character in resultPredictTop:
    plate_characters_top+= each_top_character[0]
print("TOP PLATE CHARACTERS ::::",plate_characters_top)
print("Check the error")
#ARRANGING PLATE CHARACTERS
orderedPlateTop = []
listCopyTop= License_setup.columnListTop[:]
print(listCopyTop)
for each_index_top in reversed(sorted(License_setup.columnListTop)):
    orderedPlateTop+= plate_characters_top[listCopyTop.index(each_index_top)]

print("FINAL PLATE TOP LETTERS::::",orderedPlateTop)
trueLetter = ["BA","7","9","PA"]
output2=tk.StringVar(value=" ".join(orderedPlateTop))
print(len(trueLetter))
print(len(orderedPlateTop))
cm = confusion_matrix(trueLetter,orderedPlateTop)
print(cm)



################################### BOTTOM HALF OF PLATE LETTER TREDICTIONS ############################
resultPredictBot = []
print(".....................................................")
print("...predicting bottom half plate characters...")
for each_bot_letter in License_setup.charactersBot:

    each_bot_letter = each_bot_letter.reshape(1,-1)
    result_bottom = loadModel.predict(each_bot_letter)
    resultPredictBot.append(result_bottom)

# PRINTING BOTTOM PLATE LETTERS

plate_characters_bot = ""

for each_bot_character in resultPredictBot:
    plate_characters_bot+=each_bot_character[0]

print("BOTTOM PLATE LETTERS::::",plate_characters_bot)

#ORDERING BOTTOM PLATE LETTERS
orderedPlateBot = ""
listCopyBot = License_setup.columnListBot[:]
print(listCopyBot)
for each_index_bot in sorted(License_setup.columnListBot):
    orderedPlateBot+=plate_characters_bot[listCopyBot.index(each_index_bot)]
    outputs=tk.StringVar(root1,orderedPlateBot)

print("FINAL BOTTOM CHARACTERS::::",orderedPlateBot)

frame1.pack()
tk.Label(frame1, text="Reading of number plate is :", padx=25, pady=25, font=('verdena', 22), bg='grey').grid(row=0, column=0)
tk.Label(frame1, text="Upper Plate Characters: "+output2.get(), padx=25, pady=25, font=('verdena', 16), bg='grey').grid(row=2, column=0)
lbl_pic_path1 = tk.Label(frame1, text="Lower Plate Characters: "+outputs.get(), padx=25, pady=25, font=('verdena', 16), bg='grey')
lbl_pic_path1.grid(row=3, column=0)
root1.mainloop()