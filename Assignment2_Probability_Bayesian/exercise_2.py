import random as rd

print("Exercise 2.c: ")
cnt_weath = [0,0,0]
weath = ["sunny", "cloudy", "windy"]
cnt = 0
trans = [[0.8, 1.0, 1.0],[0.4, 0.8,1.0],[0.2,0.8,1.0]]

yest = 0
today = 0
while cnt < 10000000:
	tran = trans[yest]
	p = rd.random()
	if p < tran[0]:
		today = 0
	elif p < tran[1]:
		today = 1
	else:
		today = 2
	# print(weath[today])
	cnt_weath[today] = cnt_weath[today] + 1
	cnt = cnt+1
	yest = today

print(float(cnt_weath[0])/cnt, float(cnt_weath[1])/cnt, float(cnt_weath[2])/cnt)
