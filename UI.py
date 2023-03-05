import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk , Image
# import Recognize_plate_characters
import License_setup
root = Tk()
root.title("Nepali Number Plate Character Recognition")
#root.geometry('500x500')
file1=""

frame = tk.Frame(root, bg='grey')

try:
    def test_map(x="default"):

        print("Test "+x)
        License_setup.license_main(x)
        root.destroy()


    lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25, font=('verdena',16),bg='grey')
    blb_show_pic = tk.Label(frame, bg='grey')
    entry_pic_path = tk.Entry(frame, font=('verdena',16),bg='grey')
    btn_browse = tk.Button(frame, text='Select Image', bg='grey',font=('veredna',16))
    new_button = tk.Button(frame,text="Ok", bg='grey',font=('veredna',16),command=lambda : test_map(file1))



    def selectPic():
        global img,file1
        filename = filedialog.askopenfilename(initialdir="/images", title="Select Image", filetypes=(("png images","*.png"), ("jpg images","*.jpg")))
        img = Image.open(filename)
        file1=filename
        img = img.resize((300,400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        blb_show_pic['image'] = img
        entry_pic_path.insert(0, filename)


    btn_browse['command'] = selectPic


    frame.pack()

    lbl_pic_path.grid(row=0, column=0)
    entry_pic_path.grid(row=0, column=1, padx=(0,20))
    blb_show_pic.grid(row=1, column=0, columnspan="2")
    btn_browse.grid(row=2, column=0,  columnspan="4")
    new_button.grid(row=4, column=0,  columnspan="4")

    #tk.destroy()

    root.mainloop()
except:
    print("akjsdhjkashdkasd asgd qjhasg das")