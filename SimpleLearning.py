# y = x1 + 2*x2 + 3*x3

from random import randint
from sklearn.linear_model import LinearRegression

#train data set
train_set_limit = 1000
train_set_count = 100

train_input = list()
train_output = list()

for i in range(train_set_count):
    a = randint(0, train_set_limit)
    b = randint(0, train_set_limit)
    c = randint(0, train_set_limit)
    op = a + (2*b) + (3*c)
    train_input.append([a, b, c])
    train_output.append(op)
    
for i in range(20):
    print(train_input[1], train_output[i])

#training
predictor = LinearRegression()
predictor.fit(X=train_input, y=train_output)

#prediction
x_test = [[10, 20, 30]]
outcome = predictor.predict(X=x_test)
coefficients = predictor.coef_

print('Outcome: ', outcome)
print('Coefficients: ', coefficients)

