from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlation()
        self.__inclination_degree = self.__inclination()
        self.__interception = self.__intercept()

    def __correlation(self):
        covariation = cov(self.x, self.y, bias=True)[0][1]
        variance_x = var(self.x)
        variance_y = var(self.y)
        return covariation / sqrt(variance_x * variance_y)

    def __inclination(self):
        stdx = std(self.x)
        stdy = std(self.y)
        return self.__correlation_coefficient * (stdy / stdx)

    def __intercept(self):
        meanx = mean(self.x)
        meany = mean(self.y)
        return meany - meanx * self.__inclination_degree

    def prediction(self, value):
        return self.__interception + (self.__inclination_degree * value)
    

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

lr = LinearRegression(x, y)
prediction = lr.prediction(6) #if x = 6, what is the value of y?
print(prediction)
