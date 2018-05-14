infile = open("/Users/sunjing/Desktop/machinevision-master/temp.txt", 'r')
data = infile.readlines()
data = data[0:]
svm_list = []
temp = []

for line in data:
	temp = line.strip('\n')
	temp = temp.split(' ')
	temp = [eval(x) for x in temp if x != '']
	svm_list.append(temp)
#print(temp)
infile.close()

f1 = open('/Users/sunjing/Desktop/machinevision-master/add.txt', 'r')
data1 = eval(f1.read())
for i in range(len(svm_list)):
	ele = svm_list[i][-1]
	#print(ele)
	svm_list[i][-1] = data1[i]
	svm_list[i].append(ele)
	#print(svm_list[i])

f = open('/Users/sunjing/Desktop/machinevision-master/add2.txt', 'w')
f.write(str(svm_list))
f.close()