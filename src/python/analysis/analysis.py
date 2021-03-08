from matplotlib import pyplot as plt

filepath = "L:\\pycode\\federated_learning_HE\\save\\202138040.acc"
acc = []
try:
    fp = open(filepath, 'rb')
    for line in fp.readlines():
        acc.append(float(line.decode('utf-8')))
except BaseException as e:
    print(e)

print("len:{}, acc[0-5]{}".format(len(acc), acc[0:5]))

plt.xlabel("ROUND OF TRAIN")
plt.ylabel("Accuracy")
plt.plot(acc, marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040.png")
plt.show()

plt.xlabel("ROUND OF TRAIN")
plt.ylabel("Accuracy")
plt.plot([i for i in range(0, 400)], acc[0:400], marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040-0-400.png")
plt.show()

plt.xlabel("ROUND OF TRAIN")
plt.ylabel("Accuracy")
plt.plot([i for i in range(0, 100)], acc[0:100], marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040-100.png")
plt.show()

plt.xlabel("ROUND OF TRAIN")
plt.ylabel("Accuracy")
plt.plot([i for i in range(100, 200)], acc[100:200], marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040-200.png")
plt.show()

plt.xlabel("ROUND OF TRAIN")                                               
plt.ylabel("Accuracy")
plt.plot([i for i in range(200, 300)], acc[200:300], marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040-300.png")
plt.show()

plt.xlabel("ROUND OF TRAIN")
plt.ylabel("Accuracy")
plt.plot([i for i in range(300, 400)], acc[300:400], marker=',')
plt.savefig("L:\\pycode\\federated_learning_HE\\save\\202138040-400.png")
plt.show()