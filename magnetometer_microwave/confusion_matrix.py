from openpyxl import load_workbook
from statistics import pvariance
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from classifier import s_classify

stack = []
for file in os.listdir('random_xlsx_files'):
    stack.append(file)  # stack of all files in folder to iterate through

stack_actual = []
stack_predicted = []

nothing = 'Nothing'
opened = 'Close to open'
closed = 'Open to close'

file_length = len(os.listdir('random_xlsx_files'))  # num of files in 'random_xlsx_files' folder
stack_classify = s_classify()     # imported from classifier.py, returns array of file classifications
print(stack_classify)


for i in range(0, file_length):
    print(stack_classify[i])
    print(stack[i])
    if 'o.xlsx' in stack[i] and stack_classify[i] == opened:
        stack_actual.append(0)
        stack_predicted.append(0)
    elif 'o.xlsx' in stack[i] and stack_classify[i] == closed:
        stack_actual.append(0)
        stack_predicted.append(1)
    elif 'o.xlsx' in stack[i] and stack_classify[i] == nothing:
        stack_actual.append(0)
        stack_predicted.append(2)
    elif 'c.xlsx' in stack[i] and stack_classify[i] == opened:
        stack_actual.append(1)
        stack_predicted.append(0)
    elif 'c.xlsx' in stack[i] and stack_classify[i] == closed:
        stack_actual.append(1)
        stack_predicted.append(1)
    elif 'c.xlsx' in stack[i] and stack_classify[i] == nothing:
        stack_actual.append(1)
        stack_predicted.append(2)
    elif 'n.xlsx' in stack[i] and stack_classify[i] == opened:
        stack_actual.append(2)
        stack_predicted.append(0)
    elif 'n.xlsx' in stack[i] and stack_classify[i] == closed:
        stack_actual.append(2)
        stack_predicted.append(1)
    elif 'n.xlsx' in stack[i] and stack_classify[i] == nothing:
        stack_actual.append(2)
        stack_predicted.append(2)

print(stack_actual)
print(stack_predicted)

confusion_matrix = metrics.confusion_matrix(stack_actual, stack_predicted)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=['Open', 'Closed', 'Nothing']
)
cm_display.plot()
plt.show()

#example matrix:
#y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2] #row
#y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2] #column
#print(confusion_matrix(y_actu, y_pred))