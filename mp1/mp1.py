import numpy as np
import matplotlib.pylab as plt

def getAgainst(val):
    result = 0
    if val == 0:
        result = 1
    if val == 1:
        result = 2
    return result

def checkError(competitor, me):
    if (competitor == 'P' and me == 'S') or (competitor == 'S' and me == 'R') or (competitor == 'R' and me == 'P'):
        return 1
    if competitor == me or competitor == 'Q':
        return 0
    return -1


sign = ['P', 'S', 'R']
signNum = [0, 1, 2]
probabilities = np.array([[1,0,0], [1,0,0],[1,0,0]])

against = 0
previous = 0

x = [0,]
y = [0,]

for i in range(1000):
    myVal = input("->")
    if myVal == 'P' or myVal == 'S' or myVal == 'R' or myVal == 'Q':
        print(myVal + ' vs ' + sign[against])

        x.append(x[-1] + 1)
        y.append(y[-1] + checkError(myVal, sign[against]))

        if myVal == 'P':
            probabilities[previous][0] += 1
            choice = np.random.choice(signNum,p=(probabilities[0,:]/sum(probabilities[0,:])))
            against = getAgainst(choice)
            previous = 0
        if myVal == 'S':
            probabilities[previous][1] += 1
            choice = np.random.choice(signNum, p=(probabilities[1,:]/sum(probabilities[1,:])))
            against = getAgainst(choice)
            previous = 1
        if myVal == 'R':
            probabilities[previous][2] += 1
            choice = np.random.choice(signNum, p=(probabilities[2,:]/sum(probabilities[2,:])))
            against = getAgainst(choice)
            previous = 2
        if myVal == 'Q':
            break

plt.plot(x,y)



