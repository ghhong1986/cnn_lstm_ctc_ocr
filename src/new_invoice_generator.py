# -*- coding: utf-8 -*-

# 发票数字编码生成工具
import numpy as np
import random,math
import os
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageFilter

# from gradient import genGradientImage

'''
复杂类型表现在:字体,大小,字符间隔,  相对位置,背景,旋转,扭曲等   ,字体颜色的渐变性
初始化时没有设置相关参数时,都使用默认值...
'''



def gaussian(x, a, b, c, d=0):
    return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

#
baseIm = Image.new('RGB', (304, 62))
ld = baseIm.load()
for x in range(baseIm.size[0]):
    r = int(gaussian(x, 158.8242, 201, 87.0739) + gaussian(x, 158.8242, 402, 87.0739))
    g = int(gaussian(x, 129.9851, 157.7571, 108.0298) + gaussian(x, 200.6831, 399.4535, 143.6828))
    b = int(gaussian(x, 231.3135, 206.4774, 201.5447) + gaussian(x, 17.1017, 395.8819, 39.3148))
    for y in range(baseIm.size[1]):
        ld[x, y] = (r, g, b)

def genGradientImage(width,high):
    def randbox(maxsize):
        x,y = random.randint(100,maxsize[0]-110),random.randint(0,maxsize[1]-5)
        x2,y2 = random.randint(x,maxsize[0]-100),random.randint(y,maxsize[1]-1)
        return (x,y,x2,y2)
    tmpImg = baseIm.crop(randbox(baseIm.size))
    # 可以稍许旋转
    return tmpImg.resize((width,high))


fontConstNum= {'Palatino.ttc':3,'Mshtakan.ttc':3}

def listFonts(path):
    return [ (os.path.join(path,name),fontConstNum.get(name,0)) for name in os.listdir(path) if name[:-1].endswith("tt") ]
fontArr = listFonts("../fonts")

def listBgImages(path):
    return [os.path.join(path, name) for name in os.listdir(path) if name.endswith("jpg")]

bgImgNames = listBgImages('../fpimage')

bgImgs = [ Image.open(name) for name in bgImgNames]

FACTORS = ['RELA_POS','FONT','FONT_SIZE','FONT_GAP','FADE','ROTATE','DISTORTION','TEXTLEN','BACKGROUND','FONTCOLOR']

charSet =  [c for c in u'0123456789']  #日年月校验码

class CharSet():

    def __init__(self,charset,maxlen):
        # self.randlen =  randlen
        self.char_set = charset
        self.max_size = maxlen
        self.len = len(self.char_set)

    #随机生成字串，长度固定
    #返回text,及对应的向量
    def random_text(self,randlen=False):
        text = ''
        vecs = np.zeros((self.max_size * self.len))
        if randlen:
            size = random.randint(1, self.max_size)
        else:
            size = self.max_size
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return text,vec


    # 单字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec

    # 向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if (vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        return text


def find_coeffs(pa, pb):
    '''
    参考:https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    :param pa:contains four vertices in the resulting plane.
    :param pb: is the four vertices in the current plane
    :return:
    '''
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def randStr(maxtextlen,isFix=True):
    rawtext = []
    for i in range(100):
        val = str(random.randint(0,9))
        repeat = random.randint(1,5)
        for j in range(repeat):
            rawtext.append(val)
    start  = 0

    while True:
        if isFix:
            end =  start + maxtextlen
        else:
            end = start + random.randint(maxtextlen/2,maxtextlen)
        if end < len(rawtext):
            yield  ''.join(rawtext[start:end])

        else:
            end =  end - len(rawtext)
        start = end

class PlateGenerator():

    def __init__(self,charSet,plateSize,maxLen,isFixeLen=False):
        self.charset = charSet
        self.platesize = plateSize
        self.maxlen = maxLen
        self.isfixlen = isFixeLen
        self.factors = []

        self.default_font = fontArr[0]
        self.default_font_size = 25

        self.textImage = Image.new("RGB", (400,40), (255, 255, 255))  # 白色背景

        self.charsetFun = CharSet(charSet,maxLen)

        self.charrepeatFun = randStr(maxLen,isFixeLen)

    def setFactor(self,factors):
        self.factors = factors

    def _genText(self):
        length = self.maxlen
        if self._testFactor("TEXTLEN"):
            length = random.randint(length / 2, length)
        # if not self.isfixlen:
        #     length = random.randint(length/2,length)
        ret=[]
        for i in range(length):
            ret.append(random.choice(self.charset))
        return "".join(ret)

    def _genText2(self):
        return self.charsetFun.random_text(not self.isfixlen)

    def _genText3(self):
        return self.charrepeatFun.next()

    def _testFactor(self,fact):
        return fact in self.factors

    def _getFontName(self):
        font = self.default_font
        if self._testFactor("FONT"):
            font = random.choice(fontArr)
        idx = random.randint(0,font[1])
        return font[0],idx

    def _getFontSize(self):
        fsrange= (22,30)
        if self._testFactor("FONT_SIZE"):
            fontsize = random.randint(fsrange[0],fsrange[1])
        else:
            fontsize = self.default_font_size
        return fontsize

    def _getTextPadding(self):
        '''
        获取文字前后需要添加的空格的数量
        :return:
        '''
        def randPadding():
            size = random.randint(0,3)
            return " "*size
        #     font,back
        return randPadding(),randPadding()

    # 获取绘制文字图片的大小
    def getTextImageSize(self,text,font):
        draw = ImageDraw.Draw(self.textImage)
        size = draw.textsize(text,font)
        return size

    def _paintFront(self, text):
        fname,idx = self._getFontName()
        font = ImageFont.truetype(fname, self._getFontSize(),index=idx)
        fpad,bpad = self._getTextPadding()
        ntext =  "%s%s%s"%(fpad,text,bpad)
        textsize = self.getTextImageSize(ntext,font)
        # 好像没有缩放关系
        mask = Image.new("RGBA", (textsize[0] + 15, textsize[1] + 10))  #

        draw = ImageDraw.Draw(mask)
        rw = random.randint(1, 5)
        rh = random.randint(3, 7)
        draw.text((rw, rh), ntext, font=font, fill=(255, 0, 0))
        del draw

        return mask

    def rotateImage(self,img):
        # 旋转与不旋转5:5 ,
        randDegree = [-2,-1,1,2]
        if random.randint(0,1)  == 1:
            degree= random.choice(randDegree)
            img = img.rotate(degree,expand=True)  #
        return img

    def distortImage(self,img):
        # 图形扭曲参数
        if random.randint(0,1) == 0:
            params = [1 - float(random.randint(1, 2)) / 100,
                      0,
                      0,
                      0,
                      1 - float(random.randint(1, 10)) / 100,
                      float(random.randint(1, 2)) / 500,
                      0.001,
                      float(random.randint(1, 2)) / 500
                      ]
            img = img.transform(img.size, Image.PERSPECTIVE, params)  # 创建扭曲
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强（阈值更大）

        return img

    def _getBackgroundImg(self):
        if self._testFactor("BACKGROUND"):
            return random.choice(bgImgs)
        else:
            return Image.new("RGB",(100,31),(255,255,255))


    def blendBg(self,frontImg,bgImg,maskImg):
        # TODO 重新设置相关的背景内容
        bgsize= bgImg.size
        masksize = maskImg.size
        if bgsize[0] > masksize[0] and bgsize[1] > masksize[1]:
            x = random.randint(0,bgsize[0]-masksize[0])
            y = random.randint(0,bgsize[1]-masksize[1])
            cropsize = (x,y,x+masksize[0],y+masksize[1])
            nbgImg= bgImg.crop(cropsize)
        else:
            nbgImg = bgImg.resize(maskImg.size)
        nbgImg.paste(frontImg,mask=maskImg)
        return nbgImg

    def textWithgap(self,text):
        gapsize = random.randint(0,1)
        gaptext = ' '*gapsize
        return gaptext.join(list(text))

    def randFontColor(self):
        colors = [ (0,0,0,0),(16,63,251,0),(13,153,252,0)]
        if self._testFactor("FONTCOLOR"):
            return colors[random.randint(0,2)]
        else:
            return colors[0]


    def generator(self,includeVec=False):
        text = self._genText()
        # text,vec = self._genText2()
        ntext= text
        if self._testFactor("FONT_GAP"):
            ntext = self.textWithgap(text)

        mask = self._paintFront(ntext)

        if self._testFactor("ROTATE"):
            mask = self.rotateImage(mask)
        if self._testFactor("DISTORTION"):
            mask = self.distortImage(mask)

        bg = self._getBackgroundImg()
        if self._testFactor("FADE"):
            width,high = mask.size
            front = genGradientImage(width,high)
            img = self.blendBg(front,bg,mask)
        else:

            img = self.blendBg(self.randFontColor(), bg,mask)
        imgsize = img.size
        targetwidth = imgsize[0]* self.platesize[1]*1./imgsize[1]
        targetsize = (int(targetwidth),self.platesize[1])
        if includeVec:
            return img.resize(targetsize), text,vec
        else:
            return img.resize(targetsize),text

# for TEST

def generateProperImages(file,ananoteFile,imgsize,maxlen,maxImageNum,imageFactor):
    gen = PlateGenerator(charSet, imgsize, maxlen)
    # factor = ["FONT", 'FONT_SIZE', "ROTATE"]  # ,'ROTATE',"DISTORTION","FADE" ,"FONT_GAP",
    gen.setFactor(imageFactor)
    path = '../img/' + file
    if not os.path.exists(path):
        os.mkdir(path)
    fullanno = os.path.join(path, ananoteFile)
    annotefp = open(fullanno, "w")
    for i in range(maxImageNum):
        img, text = gen.generator()
        name = '%s/%d_%s.jpg' % (path, i, text)
        img.save(name)
        print text
        annotefp.write(name + '\r\n')
    annotefp.close()



def generateImages(file,ananoteFile,size,maxlen,maxImageNum):
    gen = PlateGenerator(charSet, size, maxlen )
    factor = ["FONT", 'FONT_SIZE', "ROTATE"]  # ,'ROTATE',"DISTORTION","FADE" ,"FONT_GAP",
    gen.setFactor(factor)
    path = '../img/'+file
    if not os.path.exists(path):
        os.mkdir(path)
    fullanno = os.path.join(path,ananoteFile)
    annotefp = open(fullanno,"w")
    for i in range(maxImageNum):
        img, text = gen.generator()
        name = '%s/%d_%s.jpg' % (path, i, text)
        img.save(name)
        print text
        annotefp.write(name+'\r\n')
    annotefp.close()

def resizeHeight31(path):
    namelist = [ os.path.join(path,name) for name in  os.listdir(path) if name.endswith("jpg")]
    for name in namelist:
        img = Image.open(name)
        w,h = img.size
        targetwidth = 31.*w/h
        targetsize = (int(targetwidth), 31)
        img = img.resize((targetsize))
        img.save(name)


# FACTORS = ['RELA_POS','FONT','FONT_SIZE','FONT_GAP','FADE','ROTATE','DISTORTION','TEXTLEN','BACKGROUND','FONTCOLOR']



if __name__ == '__main__':
    defaultfacotr = ['RELA_POS','FONT',"FONT_SIZE"]  #"BACKGROUND","FONTCOLOR",'ROTATE',

    # generateImages("type1","trainiAnaote.txt",(256,31),10,50)
    # resizeHeight31('fortest')

    generateProperImages('type1','t1.txt',(256,31),10,100,defaultfacotr)

