import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

def calculate(x, y_tilde, weight_vector):
    y_dimensions = y_tilde.shape
    size = y_dimensions[0]

    result = 0

    for iter in range(size):
        exponent = y_tilde[iter] * np.transpose(weight_vector) * x[iter]
        numer = -y_tilde[iter] * x[iter]
        denom = (1 + np.exp(exponent))
        result += numer / denom

    gradient = result / size

    return gradient

def gradientDescent(x, y, stepSize, maxIterations):
 
    x_dimensions  = x.shape
    features      = x_dimensions[1]

    weight_vector = np.zeros(features)
    weight_matrix = np.zeros(shape=(features, maxIterations))

    y_size = y.shape[0]

    #create the y_tilde array
    y_tilde = np.zeros(y_size)

    for index in range(y_size):
        y_tilde[index] = 1

        if(y[index] == 0):
            y_tilde[index] = -1


    for iter in range(maxIterations):
        gradient = calculate(x, y_tilde, weight_vector)

        weight_vector = weight_vector - stepSize * gradient

        #update weight matrix
        for m_row in range (features):
            weight_matrix[m_row][iter] = weight_vector[m_row]

    return weight_matrix

# upload data
########################################## SA Heart Disease data ########################################
file = open("SAheartdata.csv", "r")

SAdata = pd.read_csv(file, sep = ",")

#convert famhist to binary indicator
Famhist = {'Present' : 1, 'Absent' : 0}
SAdata.famhist = [Famhist[item] for item in SAdata.famhist]

#create arrays for each column
d_sbp = SAdata['sbp'].array
d_tobacco = SAdata['tobacco'].array
d_ldl = SAdata['ldl'].array
d_adiposity = SAdata['adiposity'].array
d_famhist = SAdata['famhist'].array
d_typea = SAdata['typea'].array
d_obesity = SAdata['obesity'].array
d_alcohol = SAdata['alcohol'].array
d_age = SAdata['age'].array
d_chd = SAdata['chd'].array

sa_X = [d_sbp, d_tobacco, d_ldl, d_adiposity, d_famhist, d_typea, d_obesity, d_alcohol, d_age]
sa_X = np.transpose(sa_X)

sa_Y = d_chd

###########################################################################################################

################################################## spam ###################################################

file = open("spamdata.csv", "r")

spamData = np.genfromtxt(file, delimiter = ' ')

sp_Y = spamData[:, [57]]

sp_X = np.delete(spamData, 57, axis = 1)

############################################zip train######################################################

file = open("zip.train.csv", "r")

zipData = np.genfromtxt(file, delimiter = ' ')

z_Y = zipData[:, [0]]

z_X = np.delete(zipData, 0, axis = 1 )

###########################################################################################################


# begin plotting
step_size = 0.1

max_iterations = 700

def plot_error_percent(x, y, step_size, max_iterations):

    # scale inputs
    x = preprocessing.scale(x)

    #split data, train 60, test 20, validate 20
    x_train, x_test, y_train, y_test       = train_test_split(x, y,test_size=0.4,train_size=0.6)
    x_test, x_validate, y_test, y_validate = train_test_split(x_test,y_test,test_size = 0.50,train_size =0.50)

    weight_matrix = gradientDescent(x, y, step_size, max_iterations)  

    # get matrix of predicted values
    train_predict = np.matmul(x_train, weight_matrix)
    valid_predict = np.matmul(x_validate, weight_matrix)

    # round predictions
    train_predict[0.0 <= train_predict] = 1
    train_predict[0.0 > train_predict]  = 0

    valid_predict[0.0 <= valid_predict] = 1
    valid_predict[0.0 > valid_predict]  = 0

    # print counts
    print("       y           ")
    print("set            0     1")
    print(" test        " + str((y_test == 0).sum()) + "     " + str((y_test == 1).sum()))
    print(" train       " + str((y_train == 0).sum()) + "    " + str((y_test == 1).sum()))
    print(" validation  " + str((y_validate == 0).sum()) + "     " + str((y_test == 1).sum()))

    # calculate error percent
    train_error = []
    valid_error = []

    for iter in range(max_iterations):
        train_error.append(np.mean(y_train != train_predict[:,iter]))
    for iter in range(max_iterations):
        valid_error.append(np.mean(y_validate != valid_predict[:,iter]))

    # calculate minimums
    train_min = np.min(train_error)
    valid_min = np.min(valid_error)

    #train_min_y = train_error.index(train_min)
    #valid_min_y = valid_error.index(valid_min)
    
    plt.plot(train_error, label='train', c='black') 
    plt.plot(valid_error, label='valid', c='red')
    #plt.plot(xy = (train_min, train_min_y),  marker='o')
    #plt.plot(xy = (valid_min, valid_min_y), marker='o')

    plt.ylabel('percent error')
    plt.xlabel('number of iterations')
    plt.legend(loc='best')

    plt.show()
    

def plotROC(x, y):


    # scale inputs
    x = preprocessing.scale(x)

    #split data, train 60, test 20, validate 20
    x_train, x_test, y_train, y_test       = train_test_split(x, y,test_size=0.4,train_size=0.6)
    x_test, x_validate, y_test, y_validate = train_test_split(x_test,y_test,test_size = 0.50,train_size =0.50)

    weight_matrix = gradientDescent(x, y, step_size, max_iterations)  

    # print counts
    print("       y           ")
    print("set            0     1")
    print(" test        " + str((y_test == 0).sum()) + "     " + str((y_test == 1).sum()))
    print(" train       " + str((y_train == 0).sum()) + "    " + str((y_test == 1).sum()))
    print(" validation  " + str((y_validate == 0).sum()) + "     " + str((y_test == 1).sum()))

    test_predict  = np.matmul(x_test, weight_matrix)

    # calculate ROC 
    tpr, fpr, thresholds = roc_curve(y_test, test_predict[:,0], pos_label =1)

    baseline = [0, 1]

    plt.plot(baseline, baseline, c = 'violet', label = 'baseline')
    plt.plot(tpr, fpr, c = 'blue', label = 'logistic regression')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC')
    plt.legend(loc='best')
    plt.show()


######################################## Error Percent Plot ############################################

# plot for SA data
#plot_error_percent(sa_X, sa_Y, step_size, max_iterations)
#plotROC(sa_X, sa_Y)

# plot for spam data
plot_error_percent(sp_X, sp_Y, step_size, max_iterations)
plotROC(sp_X, sp_Y)

# plot for zip train data
#plot_error_percent(z_X, z_Y, step_size, max_iterations)
#plotROC(z_X, z_Y)