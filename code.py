# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
from PIL import Image
import csv
import os
import glob
import re


#这一版需要改进的问题是：确定上一页的最后到底是不是表格啊
numbers = re.compile(r'(\d+)')

picture_path = "./Pdf_Pictures"
csv_path = "./CSV"

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# 将pdf转换格式为图片：pdf2image
# 把图片保存在当前文件夹下的Pdf_Pictures文件夹中
from pdf2image import convert_from_path
def pdf_to_image2(path, file_name):
    pages = convert_from_path(path + file_name, dpi=200)
    # 创建path，包含转换出来的图片
    if not os.path.exists(picture_path + '/' + file_name[:-4]):
        os.makedirs(picture_path + '/' + file_name[:-4])
    for idx, page in enumerate(pages):
        # page.save(picture_path + 'page' + str(idx) + '.jpg', 'JPEG')
        page.save(os.path.join(picture_path + '/' + file_name[:-4], 'page' + str(idx + 1) + '.jpg'), 'JPEG')

def change_table_background(path_picture):
    # Deal with color columns
    # Load image
    original = cv2.imread(path_picture)

    # Make HSV and extract S, i.e. Saturation
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    # Save saturation just for debug
    cv2.imwrite('page_internal.png',s)

    # Load images, grayscale, Gaussian blur, Otsu's threshold
    # original = cv2.imread('1.jpg')
    internal_image = cv2.imread('page_internal.png')
    gray = cv2.cvtColor(internal_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours, filter using contour approximation + area, then extract
    # ROI using Numpy slicing and replace into original image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 1000:
            x,y,w,h = cv2.boundingRect(c)
            # print(x,y,w,h)
            ROI = internal_image[y:y+h,x:x+w]
            original[y:y+h, x:x+w] = ROI

    # 去除中间产物
    try:
        os.remove("page_internal.png")
    except:
        pass
    cv2.imwrite(path_picture, original)

def write_row(row_text, csv_writer):
    length_in_list = 0
    words_less_than_2 = 0
    empty = True
    # Clear empty string in row_text
    original_length = len(row_text)
    row_text = [string for string in row_text if string != '']
    cut_length = len(row_text)
    if(original_length - cut_length >= 2):
        empty = False


    for string in row_text:
        length_in_list += len(string.strip())

    if(len(row_text) != 0) and (length_in_list/len(row_text) > 2.5 or len(row_text) >= 3) and empty:
        if len(row_text) == 1:
            texts = row_text[0].split()
            words_less_than_2 = len([string for string in texts if len(string) <= 2])
            if words_less_than_2 < 2:
                csv_writer.writerow(row_text)
        else:
            csv_writer.writerow(row_text)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse =False
    i=0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def cell_extraction(image_path, csv_path, last_table_width,is_lastTable_little_distance):
    img=cv2.imread(image_path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 似曾相识？和上一篇中去水印的函数一样，只不过这里的Thresh方法选择的是THRESH_BINARY_INV，与THRESH_BINARY效果刚好相反，200以上转为0（黑色）,200以下转为255（白色）
    # 参数一: 140, 200
    # img_bin = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
    # 参数二: 200,255
    #[1]拿到的就是
    img_bin=cv2.threshold(img_gray,220,255,cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("Image_bin",img_bin)
    # cv2.waitKey(0)

    # define a kernel length
    kernel_length=np.array(img).shape[1]//160

    # a verticle kernel of (1 X kernel_length),which will detect all the 
    # verticle lines from the image
    verticle_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel_length))

    # a horizontal kernel of (kernel_length X 1),which will help to detect
    # all the horizontal line from the image
    hori_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_length,1))

    # a kernel of(3 X 3 )ones
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # morphological operation  to detext vertival lines from an  image
    img_temp1=cv2.erode(img_bin,verticle_kernel,iterations=3)
    verticle_lines_img=cv2.dilate(img_temp1,verticle_kernel,iterations=3)
    # cv2.imshow("verticle_lines.jpg",verticle_lines_img)
    # cv2.waitKey(0)

    #morpholigical operation to detect horizontal lines from an image
    img_temp2=cv2.erode(img_bin,hori_kernel,iterations=3)
    horizontal_lines_img=cv2.dilate(img_temp2,hori_kernel,iterations=3)
    # cv2.imshow("horizontal_lines.jpg",horizontal_lines_img)
    # cv2.waitKey(0)

    # weighting parameters,this will decide the quantity of an image to be added
    # to make a new image
    alpha=0.5
    beta=1.0-alpha

    #this function helps to add two image with specific weight parameter
    #to get a third image as summation of two image
    img_final_bin=cv2.addWeighted(verticle_lines_img,alpha,horizontal_lines_img,beta,0.0)
    img_final_bin=cv2.erode(~img_final_bin,kernel,iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, 
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("img_final_bin.jpg",img_final_bin)
    # #cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # cv2.waitKey(0)

    contours,hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    (contours,boundingBoxes)=sort_contours(contours,method="top-to-bottom")
    #idx=0

    row_text1=[]#用来保存每一行的内容
    row_text2=[]
    row_text_right=[]#用来保存可能是右边的表格的内容，当然这里还只是疑似
    is_row=True#判断是否是同一行的标记
    save_w=[]
    #记录了每一页纸上，每一个表格距离纸的下边页的距离，那么最后一个则保存了最后一张表距离底部的距离
    save_distance=[]
    contours_new=[]
    #在这里我应该记录到的是整个表格的信息，那么a就代表了整体的最左边的x坐标
    with open(csv_path+".csv",'a+',encoding='utf-8',newline='') as csvfile:
        csv_writer=csv.writer(csvfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        #当然只是对存在表格的页进行处理
        #当它里面真的只有一个单元格的时候
        #先做预处理，把不符合的先给剔除
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if(w>20 and h>20):
                contours_new.append(c)
        #print("complete")
        if(len(contours_new)==2):
            x,y,w,h=cv2.boundingRect(contours[1])
            new_img=img[y:y+h,x:x+w]
            text=pytesseract.image_to_string(new_img,config=("-l eng --oem 1 --psm 6"))
            for ch in ['|','\\','~','/','#','&']:
                if ch in text:
                    text = text.replace(ch, '')
            row_text1.append(text.replace('\n',' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
            # Debug for 鈥 character
            print("len contours=2", row_text1)
            write_row(row_text1, csv_writer)

        if(len(contours_new)>2):
            #对于每一张图，看第一个表格的x,y,w,h的值
            #拿到整张纸的信息
            page_x,page_y,page_width,page_height=cv2.boundingRect(contours[0])
            #print("打印整张纸的信息",page_x,page_y,page_width,page_height)
            #这里拿到的相当于是第一个单元格的位置信息
            first_x,first_y,first_w,first_h=cv2.boundingRect(contours_new[1])
            second_x,second_y,second_w,second_h=cv2.boundingRect(contours_new[2])
            last_y=second_y
            #用来存储的是上一个单元格的x坐标
            last_cell_x=second_x
            print("最初始的last_cell_x",last_cell_x)
            print("最初始的last_y",last_y)
            #用来暂存的是右页上的表格内容
            right_table_row=[]
            #用来标记是否存在左右分区的两个表格
            is_both_side=False
            #print("here is the information of first table ",first_x,first_y,first_w,first_h)
            #print("here is the information of second table ",second_x,second_y,second_w,second_h)
            #如果第一张表格的y坐标距离顶端太远的话直接否定其为跨页表格
            #如果第一张表格的宽度和上一张图片里最后一张表格的宽度不同，我们也认为一定不是跨页 
            #print("上一张图片中每一个表格的宽度",save_w)
            #print("上一张图片中最后一个表格的宽度",last_table_width)
            if(first_y>0.2*page_height or first_w!=last_table_width or is_lastTable_little_distance==False):
                is_spread=False
            else:
                is_spread=True
            if(is_spread==False):
                print("没有跨页")
                csv_writer.writerow(" ")
                csv_writer.writerow(" ")
            else:
                print("应该存在跨页了")

            for c in contours_new:
                x,y,w,h=cv2.boundingRect(c)
                print(x,y,w,h) 
                #我在这里调整了h
                if((h<150) and h>20) and w>20: 
                    # print("把每次的x输出",x)
                    # print("把每次的w输出",w)
                    # print("把每次的last输出",last_cell_x)
                    #print(x,y,w,h)
                    if(y==last_y):
                        is_row=True
                    else:
                        is_row=False
                    interval=y-last_y
                    last_y=y
                    new_img=img[y:y+h,x:x+w]
                    text=pytesseract.image_to_string(new_img,config=("-l eng --oem 1 --psm 6"))
                    for ch in ['|','\\','~','/','#']:
                        if ch in text:
                            text = text.replace(ch, '')
                    #只有是同一行的时候才有那么多的事情需要判断
                    if(is_row==True):
                        print("在这里输出查看绝对值",abs(x+w-last_cell_x))
                        print("这里的x",x)
                        print("这里的w",w)
                        print("这里的last_cell_x",last_cell_x)
                        if(abs(x+w-last_cell_x)<100 or abs(x+w-last_cell_x)==w):
                            #print("在这里是不分栏的")
                            #print(x,w,last_cell_x)
                            #print("在这里输出查看",abs(x+w-last_cell_x))
                            if(is_both_side==False):
                                row_text1.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                            else:
                                if(x>0.5*page_width):
                                    row_text1.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                                else:
                                    row_text2.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                        else:
                            is_both_side=True
                            row_text2.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                        last_cell_x=x
                    else:
                        last_cell_x=x
                        #遇到不同行的情况了
                        if(is_both_side==True):
                            #把row_2放好，把row_1写入csv
                            #输出来row_text1看看
                            right_table_row.append(list(reversed(row_text1)))
                            row_reversed=list(reversed(row_text2))

                            if(row_reversed!=[]):
                                print("写一行2",row_reversed)
                                write_row(row_reversed, csv_writer)
                            row_text2=[]
                            row_text1=[]
                            if(x<0.5*page_width):
                                row_text2.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                            else:
                                row_text1.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                        else:
                            row_reversed=list(reversed(row_text1))
                            print("写一行1",row_reversed)
                            write_row(row_reversed, csv_writer)
                            row_text1=[]
                            row_text2=[]
                            if(x>0.5*page_x):
                                row_text1.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                            else:
                                row_text2.append(text.replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\''))
                        if(interval>150):
                            #print("interval is ",interval)
                            print("这里应该存在第二张表格了")
                            csv_writer.writerow(" ")
                            head_img=img[y-h:y,first_x:x+w]
                            head_text=[]
                            try:
                                text = pytesseract.image_to_string(head_img,config=("-l eng --oem 1 --psm 6")).replace('\n', ' ').replace('“','\'').replace('‘', '\'').replace('’','\'').replace('”', '\'')
                                for ch in ['|','\\','~','/','#']:
                                    if ch in text:
                                        text = text.replace(ch, '')
                                head_text.append(text)
                                
                            except:
                                pass
                            #把所谓的表头输出来看一遍
                            print("print head text here",head_text)
                            write_row(head_text, csv_writer)
                            #这块要怎么处理还是一个问题
                            # print("len",len(head_text))cv
                            # if(len(head_text)):
                            #     print("没有表头信息")
                            # else:
                            #     print(head_text)
                            #     csv_writer.writerow(head_text)
                            
                #对于那些h大于150的就是图片本身和每一张表格整体的数据
                if(h>150):
                    save_w.append(w)
                    distance=page_height-(y+h)
                    save_distance.append(distance)
            
            #这里是为了写入表格的最后一行的信息
            #print("输出来row_text1看看",row_text1)
            if(row_text1!=[]):
                row_reversed=list(reversed(row_text1))
                write_row(row_reversed, csv_writer)
                print("最后",row_reversed)
                
            row_reversed=list(reversed(row_text2))
            if(row_reversed!=[]):
                print("last line:",row_reversed)
                write_row(row_reversed, csv_writer)
            if(len(right_table_row)!=0):
                #rint("输出看看",right_table_row)
                csv_writer.writerow(" ")
                for row in right_table_row:
                    if(row!=[]):
                        print("写一行3",row)
                        write_row(row, csv_writer)
            

    #有可能该页面没有任何的表格
    if(len(save_w)!=0 and len(save_distance)!=0):
        last_table_width=save_w[-1]
        if(save_distance[-1]<0.2*page_height):
            is_lastTable_little_distance=True
    else:
        last_table_width=0
        is_lastTable_little_distance=False
    #把情况输出看看
    if(is_lastTable_little_distance==True):
        print("这张纸的表格是到了最后了")
    else:
        print("表格没有到最后")
    return last_table_width,is_lastTable_little_distance


def extract_pdfs(pdf_path):
    # 准备工作1：创建一个叫CSV的文件夹
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    # 准备工作2：对pdf每一页进行提取，转换为图片格式
    for pdf_file in os.listdir(pdf_path):
        if pdf_file.endswith(".pdf"):
            pdf_to_image2(pdf_path + '/', pdf_file)

        # 形式：./Pdf_Pictures/pdf的名称/第几页.jpg
        if str(pdf_file) != ".DS_Store":
            img_path = picture_path + '/' + pdf_file[:-4]
            # 设一个标志,判断是否可能存在跨页
            is_spread=True
            # 设置一个参数，记录的是前一页最后一张表格是否已经是整张纸的底部
            is_lastTable_little_distance=False
            #设置一个参数，记录前一页最后一张表格的宽度
            last_table_width=0
            # 对图片排序，进行提取
            for image in sorted(glob.glob(img_path + '/' + '*.jpg'), key=numericalSort):
                # do something here
                #change_table_background(image)
                print("image",image)
                csv_filename = csv_path + '/' + pdf_file[:-4]
                print(csv_filename)
                #box_extraction(image, csv_path + '/' + pdf_file[:-4])
                last_table_width,is_lastTable_little_distance= cell_extraction(image, csv_filename, last_table_width, is_lastTable_little_distance)
            


extract_pdfs("./Pdfs")