import cv2
import pytesseract
print(pytesseract.pytesseract.tesseract_cmd)

img=cv2.imread('1.jpg')
text=pytesseract.image_to_string(img)
print(text)