# -*- coding: utf-8 -*-
import os

class Article():
    def __init__(self, name,parser):
        self.name = name
        self.lines = []
        self.parser = parser

    def appendTitle(self, level, title):
        pass

    def appendLine(self,line):
        self.lines.append(self.parser(line))

    def save(self, filename):
        f = open(filename, "w")
        f.writelines(self.lines)
        f.close()


class HtmlArticle(Article):
    htmlformat = '''<!DOCTYPE html>
       <html lang="en">
       <head>
           <meta charset="UTF-8">
           <title>%s</title>
       </head>
       <body>
         %s
       </body>
       </html>
       '''

    def appendTitle(self, level, title):
        ft = "<h%d> %s </h%d>" % (level, title, level)
        self.lines.append(ft)

    def save(self, filename):
        f = open(filename, "w")
        body = '<p/>\n'.join(self.lines)
        full = self.htmlformat%(self.name,body);
        f.write(full)
        f.close()


stImgPath = '/Users/honggh/Desktop/scale2'
def standParserMaker(imgpath):
    def lineParser(text):
        items = text.split(' ')
        filename = items[1]
        target = items[3]
        fullname = os.path.join(imgpath,filename)
        imgtextFormat = '<img src="%s"  alt="%s" />  %s'
        return imgtextFormat%(fullname,target,target)
    return lineParser

print standParserMaker(stImgPath)('W 62_20170730.jpg ---> 201740711301')



##  W 62_20170730.jpg ---> 201740711301
wrongRes = '/Users/honggh/logs/ocr_image/wrong.txt'
htmlMaker =  HtmlArticle("Wrong",standParserMaker(stImgPath))

file = open(wrongRes)
text = file.readline()
while text != "":
    htmlMaker.appendLine(text.strip('\n'))
    text = file.readline()

htmlMaker.save("../wrong.html")